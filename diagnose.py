"""
诊断脚本：对比模型在不同测试集上的表现
用法: python diagnose.py --model_path <path_to_model.pth> --gpu_ids 0
"""
import torch, os, argparse
from networks.resnet import resnet50
from validate import validate
from options.test_options import TestOptions


def test_one(model, dataroot, name, batch_size=64, no_crop=True):
    """Run validate on a single dataset."""
    # 构造 options，直接用 argparse 解析，绕过 sys.argv
    opt_obj = TestOptions()
    opt_obj.gather_options()  # 初始化 parser
    opt_obj.parser.prog = 'diagnose'  # 避免 sys.argv[0] 干扰

    opt_args = ['--dataroot', dataroot,
                '--batch_size', str(batch_size),
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
    # 模型配置（默认优化版，基线模型用 --no_multi_scale --no_sobel --no_tkp）
    parser.add_argument('--no_multi_scale', action='store_true')
    parser.add_argument('--no_sobel', action='store_true')
    parser.add_argument('--no_tkp', action='store_true')
    parser.add_argument('--use_rgb_branch', action='store_true')
    args = parser.parse_args()

    multi_scale = not args.no_multi_scale
    use_sobel   = not args.no_sobel
    use_tkp     = not args.no_tkp

    print(f"Model: {args.model_path}")
    print(f"Config: multi_scale={multi_scale} sobel={use_sobel} tkp={use_tkp} rgb_branch={args.use_rgb_branch}")
    print()

    # 加载模型
    model = resnet50(num_classes=1,
                     multi_scale=multi_scale,
                     use_sobel=use_sobel,
                     use_tkp=use_tkp,
                     tkp_k=args.tpk_k,
                     use_rgb_branch=args.use_rgb_branch)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    print("=" * 75)

    # ── 测试集列表 ──────────────────────────────
    tests = [
        # (name, dataroot, no_crop)
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
            r = test_one(model, dataroot, name, args.batch_size, no_crop)
            results.append((name, *r))
        except Exception as e:
            print(f"  {name:<25s}  ERROR: {e}")

    print("=" * 75)
    print()
    print("Summary:")
    print(f"{'Test set':<30s} {'Acc':>6s} {'AP':>6s} {'Real':>6s} {'Fake':>6s}")
    print("-" * 54)
    for name, acc, ap, r_acc, f_acc in results:
        print(f"{name:<30s} {acc*100:5.1f}% {ap*100:5.1f}% {r_acc*100:5.1f}% {f_acc*100:5.1f}%")


if __name__ == '__main__':
    main()
