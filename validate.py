import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    # compat with sklearn >= 1.6
    try:
        ap = average_precision_score(y_true, y_pred)
    except ValueError:
        ap = average_precision_score(y_true, y_pred.reshape(-1, 1))
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    state_dict = torch.load(opt.model_path, map_location='cpu')
    raw_state = state_dict.get('model', state_dict)

    # Auto-detect architecture flags from checkpoint
    use_post_bn = 'post_pool_bn.weight' in raw_state
    multi_scale = 'conv1.weight' in raw_state and raw_state['conv1.weight'].shape[1] == 9
    use_sobel   = 'sobel_x' in raw_state
    use_tkp     = hasattr(opt, 'use_tkp') and getattr(opt, 'use_tkp', False)

    model = resnet50(num_classes=1,
                     use_attn_pool=getattr(opt, 'use_attn_pool', False),
                     multi_scale=multi_scale,
                     use_sobel=use_sobel,
                     use_tkp=use_tkp,
                     tkp_k=getattr(opt, 'tkp_k', 5),
                     full_layers=True,
                     use_post_bn=use_post_bn)
    model.load_state_dict(raw_state)
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
