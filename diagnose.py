"""
诊断脚本：对比模型在不同测试集上的表现
用法: python diagnose.py --model_path <path_to_model.pth> --gpu_ids 0
TTA:   python diagnose.py --model_path <path> --tta
"""
import torch, os, argparse, numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import datasets
from sklearn.metrics import average_precision_score, accuracy_score
from networks.resnet import resnet50
from validate import validate
from options.test_options import TestOptions


def validate_tta(model, opt):
    """Validate with Test-Time Augmentation: 5-crop + flip = 10 views per image."""
    crop_size = opt.cropSize
    img_size  = opt.loadSize

    # Crop positions
    dh, dw = img_size - crop_size, img_size - crop_size
    positions = [
        (0, 0),           # top-left
        (0, dw),          # top-right
        (dh, 0),          # bottom-left
        (dh, dw),         # bottom-right
        (dh//2, dw//2),   # center
    ]

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Check structure: ImageFolder expects class subdirs
    root = opt.dataroot
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataroot not found: {root}")

    # Auto-detect if root has class subdirs, else use as-is
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if '0_real' in subdirs and '1_fake' in subdirs:
        pass  # root is a valid ImageFolder directory
    else:
        raise ValueError(f"Expected 0_real/ and 1_fake/ under {root}, got {subdirs}")

    dataset = datasets.ImageFolder(root, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                          shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda()
            B, C, H, W = imgs.shape

            # Collect 10 views per image
            all_logits = []
            for (top, left) in positions:
                crop = imgs[:, :, top:top+crop_size, left:left+crop_size]
                logit = model(crop)                         # original
                logit_flip = model(TF.hflip(crop))           # flipped
                all_logits.append(logit)
                all_logits.append(logit_flip)

            # Average over 10 views → sigmoid
            avg_logit = torch.stack(all_logits, dim=0).mean(dim=0)
            probs = avg_logit.sigmoid().flatten().tolist()

            y_pred.extend(probs)
            y_true.extend(labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc  = accuracy_score(y_true, y_pred > 0.5)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    ap   = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


def test_one(model, dataroot, name, batch_size=64, no_crop=True, tta=False):
    """Run validate on a single dataset."""
    if tta:
        # construct opts for TTA
        opt_obj = TestOptions()
        opt_obj.gather_options()
        opt_obj.parser.prog = 'diagnose'
        opt_args = ['--dataroot', dataroot, '--batch_size', str(min(batch_size, 16)),
                    '--gpu_ids', '0']
        parsed = opt_obj.parser.parse_args(opt_args)
        parsed.isTrain = False
        parsed.classes = []
        parsed.loadSize = 256
        parsed.cropSize = 224

        acc, ap, r_acc, f_acc, _, _ = validate_tta(model, parsed)
    else:
        opt_obj = TestOptions()
        opt_obj.gather_options()
        opt_obj.parser.prog = 'diagnose'
        opt_args = ['--dataroot', dataroot, '--batch_size', str(batch_size),
                    '--gpu_ids', '0']
        if no_crop:
            opt_args.append('--no_crop')
        parsed = opt_obj.parser.parse_args(opt_args)
        parsed.isTrain = False
        parsed.classes = []

        acc, ap, r_acc, f_acc, _, _ = validate(model, parsed)

    print(f"  {name:<25s}  acc={acc*100:5.1f}%  ap={ap*100:5.1f}%  "
          f"real={r_acc*100:5.1f}%  fake={f_acc*100:5.1f}%")
    return acc, ap, r_acc, f_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--gpu_ids', default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--tpk_k', type=int, default=5)
    parser.add_argument('--tta', action='store_true', help='enable test-time augmentation')
    parser.add_argument('--no_multi_scale', action='store_true')
    parser.add_argument('--no_sobel', action='store_true')
    parser.add_argument('--no_tkp', action='store_true')
    parser.add_argument('--no_layer34', action='store_true', help='use 2-layer ResNet')
    args = parser.parse_args()

    multi_scale = not args.no_multi_scale
    use_sobel   = not args.no_sobel
    use_tkp     = not args.no_tkp
    full_layers = not args.no_layer34

    print(f"Model: {args.model_path}")
    print(f"Config: multi_scale={multi_scale} sobel={use_sobel} tkp={use_tkp} full_layers={full_layers} tta={args.tta}")
    print()

    model = resnet50(num_classes=1,
                     multi_scale=multi_scale,
                     use_sobel=use_sobel,
                     use_tkp=use_tkp,
                     tkp_k=args.tpk_k,
                     full_layers=full_layers)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    print("=" * 75)

    tests = [
        ("ForenSynths_progan", "./dataset/ForenSynths_train_val/test/progan", True),
        ("NPR_Test_jpeg_q95",  "./NPR_Test_jpeg_q95/my_first_test",      True),
        ("NPR_Test_jpeg_q98",  "./NPR_Test_jpeg_q98/my_first_test",      True),
        ("NPR_Test_clean",     "./NPR_Test/my_first_test",               True),
    ]

    results = []
    for name, dataroot, no_crop in tests:
        if not os.path.exists(dataroot):
            print(f"  {name:<25s}  SKIP (path not found: {dataroot})")
            continue
        try:
            r = test_one(model, dataroot, name, args.batch_size, no_crop, tta=args.tta)
            results.append((name, *r))
        except Exception as e:
            print(f"  {name:<25s}  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("=" * 75)
    print()
    print("Summary:")
    print(f"{'Test set':<30s} {'Acc':>6s} {'AP':>6s} {'Real':>6s} {'Fake':>6s}")
    print("-" * 54)
    for name, acc, ap, r_acc, f_acc in results:
        print(f"{name:<30s} {acc*100:5.1f}% {ap*100:5.1f}% {r_acc*100:5.1f}% {f_acc*100:5.1f}%")

if __name__ == '__main__':
    main()
