"""Held-out autoregressive dynamics evaluation on actions actually executed in MuJoCo."""
import torch

from .model import quat_matrix, _rotation_error


def _phase_masks(states, contacts, targets, events):
    """Endpoint phase, with precedence for landing, then slip, held, and empty."""
    device = states.device
    count = contacts.shape[0]
    hand = contacts[:, 2:4].gt(.5).any(-1)
    expected = targets[1:, 37:39].gt(.5).any(-1)
    if events is not None:
        if len(events) != count:
            raise ValueError("one event is required for each transition")
        def flag(key):
            return torch.tensor([bool(event.get(key, False)) for event in events], device=device)
        support = flag("supported")
        lifted = flag("lifted")
        landing = flag("drop") | (lifted & support & ~torch.cat((support[:1], support[:-1])))
    else:
        lifted = hand.long().cummax(0).values.bool()
        landing = torch.zeros_like(hand)
    # Contact loss after carrying while a grasp is still required distinguishes
    # slip from the intended release onto a table.
    slip = lifted & expected & ~hand & ~landing
    held = hand & ~landing
    return {"all":torch.ones_like(hand), "held":held, "slip":slip,
            "landing":landing, "empty":~(held | slip | landing)}


@torch.no_grad()
def validate_dynamics(wm, states, actions, contexts, contacts, target_frames,
                      terminals, *, stride=4, max_windows=128, events=None):
    """Evaluate one continuous episode. Metrics use the true terminal endpoint.

    A window may end in a terminal state, but may never include a reset. The
    simulator baseline predicts that the box stays at its measured start pose.
    Stage counts reflect sampled windows, not episode success or duration.
    """
    count = actions.shape[0]
    if (len(states) != count+1 or len(target_frames) != count+1
            or len(contacts) != count or len(terminals) != count):
        raise ValueError("inconsistent validation state/transition counts")
    if stride <= 0 or max_windows <= 0:
        raise ValueError("validation strides and budgets must be positive")
    if not count:
        return {}
    if contexts.ndim == 1:
        contexts = contexts.expand(count, -1)
    if len(contexts) != count:
        raise ValueError("contexts must be per-episode or per-transition")
    masks = _phase_masks(states, contacts, target_frames, events)
    device = states.device
    prefix = torch.cat((torch.zeros(1,device=device,dtype=torch.long),
                        terminals.reshape(count).long().cumsum(0)))
    result = {}
    for horizon in (1,8,32):
        if count < horizon:
            continue
        # Retain rare landing/release observations in addition to regularly
        # spaced starts, so a one-frame collision can contribute a measurement.
        base = torch.arange(0, count-horizon+1, stride, device=device)
        landing_end = masks["landing"].nonzero().flatten() - horizon + 1
        terminal_end = terminals.reshape(-1).nonzero().flatten() - horizon + 1
        starts = torch.unique(torch.cat((base,landing_end,terminal_end)))
        starts = starts[starts>=0]
        starts = starts[prefix[starts+horizon-1] == prefix[starts]]
        if len(starts)>max_windows:
            which = torch.linspace(0,len(starts)-1, max_windows,device=device).round().long()
            starts = starts[which]
        if not len(starts):
            continue
        end = starts + horizon
        pred = states[starts]
        for offset in range(horizon):
            index = starts+offset
            pred, logits = wm(pred, actions[index], contexts[index])
        truth = states[end]
        metrics = {
            "box_position_m": (pred[:,71:74]-truth[:,71:74]).norm(dim=-1),
            "stationary_box_position_m": (states[starts,71:74]-truth[:,71:74]).norm(dim=-1),
            "box_rotation_rad": _rotation_error(quat_matrix(pred[:,74:78]),quat_matrix(truth[:,74:78])),
            "root_position_m": (pred[:,:3]-truth[:,:3]).norm(dim=-1),
            "root_rotation_rad": _rotation_error(quat_matrix(pred[:,3:7]),quat_matrix(truth[:,3:7])),
            "joint_mae_rad": (pred[:,7:36]-truth[:,7:36]).abs().mean(-1),
            "contact_accuracy": ((logits[:,:4]>0)==(contacts[end-1,:4]>.5)).float().mean(-1),
            "support_accuracy": ((logits[:,4:]>0)==(contacts[end-1,4:]>.5)).float().mean(-1),
        }
        for phase, rows in masks.items():
            selected = rows[end-1]
            number = int(selected.sum().item())
            key = f"h{horizon:02d}/{phase}"
            result[key+"/windows"] = number
            if number:
                for name, values in metrics.items():
                    result[f"{key}/{name}"] = values[selected].mean().item()
    return result


def aggregate_dynamics(episodes):
    """Weighted averages across episodes for each horizon/phase."""
    sums, counts = {}, {}
    for episode in episodes:
        for key, value in episode.items():
            group, metric = key.rsplit("/",1)
            if metric=="windows":
                counts[group] = counts.get(group,0)+value
            else:
                sums[key] = sums.get(key,0.)+value*episode[group+"/windows"]
    return {**{group+"/windows":v for group,v in counts.items()},
            **{key:v/counts[key.rsplit("/",1)[0]] for key,v in sums.items()}}
