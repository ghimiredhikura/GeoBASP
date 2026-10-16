"""Nearest aligned MAC target for the post-training SIRFP adaptation."""
import copy
import torch
import torch_pruning as tp
from nepal_cnn.pruning.architecture import build_meta_pruner
from nepal_cnn.pruning.sirfp_convnext import protected_shapes, validate_structure

def prune_to_target(model, importance, example, base_macs, target, protocol, history, logger):
    steps = []
    convnext = protocol["architecture"] == "unet_convnextv2_tiny"
    protected = protected_shapes(model) if convnext else None
    current_macs, _ = tp.utils.count_ops_and_params(model, example)
    achieved = 1 - current_macs / base_macs
    for index in range(1, protocol["maximum_microsteps_per_target"] + 1):
        if achieved >= target:
            break
        # CPU snapshot uses the same nearest-target decision as the controls.
        previous = copy.deepcopy(model.cpu())
        model = model.to(example.device)
        previous_macs, previous_achieved = int(current_macs), achieved
        pruner, _ = build_meta_pruner(model, example, protocol["architecture"], importance,
                                     protocol["micro_channel_pruning_ratio"])
        for group in pruner.step(interactive=True):
            importance.apply_removal(group)
            group.prune()
        step_history = copy.deepcopy(pruner.pruning_history())
        model.eval()
        with torch.inference_mode():
            output = model(example)
        if output.shape != (1, 1, 512, 512) or not torch.isfinite(output).all():
            raise RuntimeError("Invalid output after physical channel removal")
        if convnext:
            validate_structure(model, importance.graphs, protected)
        current_macs, parameters = tp.utils.count_ops_and_params(model, example)
        achieved = 1 - current_macs / base_macs
        if current_macs >= previous_macs:
            raise RuntimeError("Micro-step did not reduce MACs")
        accepted = achieved <= target or abs(achieved - target) <= abs(previous_achieved - target)
        steps.append({"micro_step": index, "previous_macs": previous_macs, "candidate_macs": int(current_macs),
                      "candidate_parameters": int(parameters), "candidate_mac_reduction": achieved,
                      "accepted": accepted})
        if accepted:
            history.extend(step_history)
            del previous
        else:
            del model
            model = previous.to(example.device)
            current_macs, achieved = previous_macs, previous_achieved
        logger.info("Target %.0f%% | micro-step %02d | candidate %.2f%% | %s", target * 100,
                    index, steps[-1]["candidate_mac_reduction"] * 100, "accepted" if accepted else "kept previous")
        del pruner, output
        torch.cuda.empty_cache()
        if not accepted or achieved >= target:
            break
    else:
        raise RuntimeError("MAC target not reached within the frozen micro-step budget")
    if abs(achieved - target) > protocol["maximum_target_absolute_error"]:
        raise RuntimeError(f"Achieved MAC reduction {achieved:.4f} is outside target tolerance")
    return model, achieved, history, steps

