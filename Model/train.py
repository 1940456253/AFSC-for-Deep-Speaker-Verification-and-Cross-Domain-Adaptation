"""Train at epoch granularity; checkpoints include AFSC, optimizer and RNG state."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from Dataset.data import SpeakerDataset
from Features.frontend import build_extractor
from Model.system import SpeakerSystem, CosineClassifier, angular_margin_loss


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def stage_for(epoch, cfg):
    return 'joint' if cfg['feature'] == 'afsc' and epoch <= cfg['joint_epochs'] else 'frozen'


def make_optimizer(model, classifier, cfg, stage):
    model.freeze_frontend(stage != 'joint')
    groups = [{'params': list(model.encoder.parameters()) + list(classifier.parameters()),
               'lr_scale': 1.0, 'weight_decay': cfg['weight_decay']}]
    if stage == 'joint':
        groups.append({'params': list(model.frontend.parameters()),
                       'lr_scale': cfg['gap_lr_multiplier'], 'weight_decay': 0.0})
    return torch.optim.SGD(groups, lr=cfg['learning_rate'], momentum=cfg['momentum'],
                           nesterov=cfg['momentum'] > 0)


def learning_rate(progress, cfg):
    total = cfg.get('lr_decay_epochs', cfg['joint_epochs'] + cfg['frozen_epochs'])
    low, high, warm = cfg['min_learning_rate'], cfg['learning_rate'], cfg['warmup_epochs']
    if warm > 0 and progress < warm:
        return low + (high - low) * progress / warm
    if progress >= total:
        return low
    return low + 0.5 * (high - low) * (1 + math.cos(math.pi * (progress - warm) / (total - warm)))


def current_margin(progress, cfg):
    if cfg['margin_schedule'] == 'constant':
        return cfg['margin']
    start, end = cfg['margin_start_epoch'], cfg['margin_end_epoch']
    if progress < start:
        return 0.0
    if progress >= end:
        return cfg['margin']
    return cfg['margin'] * (1 - math.exp((progress - start) / (end - start) * math.log(1e-3 / 1.000001)))


def rng_state(generator):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['loader'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_config(cfg):
    if cfg['sample_rate'] != 16000:
        raise ValueError('This front end requires 16000 Hz')
    if cfg['joint_epochs'] < 0 or cfg['frozen_epochs'] < 0 or cfg['joint_epochs'] + cfg['frozen_epochs'] < 1:
        raise ValueError('Invalid epoch counts')
    total = cfg['joint_epochs'] + cfg['frozen_epochs']
    if not 0 <= cfg['warmup_epochs'] < cfg.get('lr_decay_epochs', total):
        raise ValueError('warmup_epochs must be >= 0 and less than total epochs')
    if cfg['batch_size'] < 2:
        raise ValueError('ECAPA batch normalization requires training batch_size >= 2')
    if not 0 <= cfg['min_learning_rate'] <= cfg['learning_rate']:
        raise ValueError('Invalid learning rate range')
    if cfg['margin_schedule'] not in ('constant', 'ramp'):
        raise ValueError('margin_schedule must be constant or ramp')
    if cfg['margin_schedule'] == 'ramp' and cfg['margin_end_epoch'] <= cfg['margin_start_epoch']:
        raise ValueError('Invalid margin ramp interval')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='Model/config.yaml')
    p.add_argument('--train-csv', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', help='Resume our checkpoint at the next epoch')
    p.add_argument('--feature', choices=['afsc', 'fbank', 'mfcc'])
    p.add_argument('--stop-after-epoch', type=int, help='Stop early without changing the full schedule')
    args = p.parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.feature:
        cfg['feature'] = args.feature
    validate_config(cfg)
    device = torch.device(args.device)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(cfg['seed'])
    dataset = SpeakerDataset(args.train_csv, build_extractor(cfg['feature']),
                             cfg['crop_seconds'], cfg['sample_rate'])
    if len(dataset.speakers) < 2:
        raise ValueError('Need at least two training speakers')
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'], drop_last=True,
                        worker_init_fn=seed_worker, generator=generator,
                        pin_memory=device.type == 'cuda')
    if not len(loader):
        raise ValueError('Training set smaller than batch_size')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Output already contains last.pt; use --resume or a new directory')
    model = SpeakerSystem(cfg).to(device)
    classifier = CosineClassifier(cfg['embedding_dim'], len(dataset.speakers)).to(device)
    epoch_start = 1
    stage = stage_for(1, cfg)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != cfg or checkpoint['speakers'] != dataset.speakers:
            raise ValueError('Resume requires identical configuration and speaker mapping')
        if checkpoint['manifest_rows'] != dataset.rows:
            raise ValueError('Resume requires the same manifest entries')
        model.load_state_dict(checkpoint['model'])
        classifier.load_state_dict(checkpoint['classifier'])
        epoch_start = checkpoint['epoch'] + 1
        stage = checkpoint['stage']
    optimizer_stage = ('joint' if checkpoint and len(checkpoint['optimizer']['param_groups']) == 2 else stage)
    optimizer = make_optimizer(model, classifier, cfg, optimizer_stage)
    model.freeze_frontend(stage != 'joint')
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        restore_rng(checkpoint['rng'], generator)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / 'speakers.json').write_text(json.dumps(dataset.speakers, indent=2))
    total = cfg['joint_epochs'] + cfg['frozen_epochs']
    end = min(total, args.stop_after_epoch) if args.stop_after_epoch else total
    for epoch in range(epoch_start, end + 1):
        next_stage = stage_for(epoch, cfg)
        if next_stage != stage:
            model.freeze_frontend(True)
            if cfg['reset_optimizer_at_freeze']:
                optimizer = make_optimizer(model, classifier, cfg, next_stage)
            stage = next_stage
        model.train(); classifier.train()
        losses, correct, count = 0.0, 0, 0
        for step, (features, labels) in enumerate(loader):
            progress = epoch - 1 + step / len(loader)
            base_lr = learning_rate(progress, cfg)
            for group in optimizer.param_groups:
                group['lr'] = base_lr * group['lr_scale']
            margin = current_margin(progress, cfg)
            features, labels = features.to(device), labels.to(device)
            cosine = classifier(model(features))
            loss = angular_margin_loss(cosine, labels, cfg['scale'], margin)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses += loss.item() * labels.numel()
            correct += int((cosine.argmax(1) == labels).sum())
            count += labels.numel()
            if step % cfg['log_every'] == 0:
                print(f'epoch={epoch}/{total} stage={stage} step={step}/{len(loader)} loss={loss.item():.4f} lr={base_lr:.6g} margin={margin:.4f}', flush=True)
        record = {'epoch': epoch, 'stage': stage, 'loss': losses / count, 'accuracy': correct / count,
                  'lr': base_lr, 'margin': margin, 'samples': count}
        with (out / 'train.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        payload = {'format_version': 1, 'config': cfg, 'epoch': epoch, 'stage': stage,
                   'model': model.state_dict(), 'classifier': classifier.state_dict(),
                   'optimizer': optimizer.state_dict(), 'speakers': dataset.speakers,
                   'manifest_rows': dataset.rows, 'rng': rng_state(generator)}
        save_checkpoint(out / f'epoch_{epoch:03d}.pt', payload)
        save_checkpoint(out / 'last.pt', payload)
        if cfg['feature'] == 'afsc':
            points = model.frontend.get_bin_points().detach().cpu().tolist()
            (out / f'frequency_points_{epoch:03d}.json').write_text(json.dumps({'bin_points': points, 'hz': [v * 31.25 for v in points]}, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
