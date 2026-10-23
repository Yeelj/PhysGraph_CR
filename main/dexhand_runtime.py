"""Named joint ordering at the policy/Isaac Gym boundary.

This module deliberately has no simulator imports so ordering checks can run
before expensive environments are created.
"""


def joint_order_indices(policy_names, simulation_names):
    """Return gather indices for simulation->policy and policy->simulation."""

    policy_names = tuple(policy_names)
    simulation_names = tuple(simulation_names)
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("Policy DOF names must be unique")
    if len(set(simulation_names)) != len(simulation_names):
        raise ValueError("Simulation DOF names must be unique")
    if set(policy_names) != set(simulation_names):
        raise ValueError(
            "Hand asset DOFs differ from the policy specification: "
            f"missing={sorted(set(policy_names) - set(simulation_names))}, "
            f"unexpected={sorted(set(simulation_names) - set(policy_names))}"
        )
    return (
        tuple(simulation_names.index(name) for name in policy_names),
        tuple(policy_names.index(name) for name in simulation_names),
    )


def validate_direct_control(config):
    """Reject unsupported control layouts before constructing a policy/sim."""

    if config.get("policyMode", "direct") != "direct":
        raise ValueError("The bimanual Transformer supports policyMode=direct")
    if config.get("bimanual_mode", "united") != "united":
        raise ValueError("The direct bimanual Transformer requires bimanual_mode=united")
    if config.get("usePIDControl", False) or config.get("useQuatRot", False):
        raise ValueError(
            "The direct bimanual Transformer uses 6 wrist controls per hand; "
            "usePIDControl and useQuatRot must be false"
        )
