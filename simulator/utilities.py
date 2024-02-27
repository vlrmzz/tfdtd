def print_setup_info(simulation):
    print("Simulation setup information:")
    print(f"nx: {simulation.nx}")
    print(f"ny: {simulation.ny}")
    print(f"time_steps: {simulation.time_steps}")
    print(f"dx: {simulation.dx}")
    print(f"dy: {simulation.dy}")
    print(f"dt: {simulation.dt}")
    print(f"use_pml: {simulation.use_pml}")
    if simulation.use_pml:
        print(f"pml_thickness: {simulation.pml_thickness}")
    print(f"c: {simulation.c}")
    print(f"epsilon_0: {simulation.eps_0}")
    print(f"mu_0: {simulation.mu_0}")
    for i, source in enumerate(simulation.sources):
        print(f"Source {i}: {source}")
    for i, detector in enumerate(simulation.detectors):
        print(f"Detector {i}: {detector}")