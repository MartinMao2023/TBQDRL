"""Trajectory relocation optimization (Stage 3).

Given a critic and a set of rollout trajectories (collected under some
preference), decide which trajectory segments to relocate to which new
preference (goal) so as to maximize the relocation advantage, and produce
the relabeled trajectories together with a per-sample include/exclude flag.
"""

from data_struct.relocation_transitions import MORelocationTransition


def relocate(
    transitions: MORelocationTransition,
    critic_network,
    critic_params,
    config: dict,
    key,
) -> MORelocationTransition:
    """[placeholder] Relabel trajectories to new goals by maximizing the
    relocation advantage, and select which segments to keep.

    Inputs:
        transitions     MORelocationTransition (may come from anywhere);
                        trajectory structure is preserved.
        critic_network  value network used to evaluate advantages.
        critic_params   critic parameters.
        config          relocation hyperparameters.
        key             RNGKey.

    Returns (placeholder): the input ``transitions`` unchanged.

    TODO:
        * for each candidate preference, recompute the scalar reward as
          ``sum(mo_rewards * preference)`` and the value
          ``V(obs, z_new)`` with ``z_new = concat(last_action, preference)``;
        * compute the relocation advantage and maximize it over candidate
          preferences to pick the new goal per segment;
        * mark trajectories that contain a done or truncation as
          include=False (passed in but always discarded);
        * return the relabeled trajectories plus a per-sample include flag
          (the output size/shape will differ from the input — only selected
          segments are kept — and will be rearranged to a designed shape).
    """
    return transitions
