def print_setup_info(simulation):
    print("Simulation setup information:")
    print(f"nx: {simulation.nx}")
    print(f"ny: {simulation.ny}")
    print(f"time_steps: {simulation.time_steps}")
    print(f"time: {simulation.time}")
    print(f"dx: {simulation.dx}")
    print(f"dy: {simulation.dy}")
    print(f"dt: {simulation.dt}")
    print(f"use_pml: {simulation.use_pml}")
    if simulation.use_pml:
        print(f"pml_thickness: {simulation.pml_thickness}")
    print(f"use_tfsf: {simulation.use_tfsf}")
    if simulation.use_tfsf:
        print(f"tfsf_thickness: {simulation.tfsf_thickness}")
    print(f"polarization: {simulation.polarization}")
    print(f"function: {simulation.function}")
    if simulation.function:
        print(f"frequency: {simulation.frequency}")
    print(f"source_type: {simulation.source_type}")
    if simulation.source_type == 'line_source':
        print(f"line x: {simulation.line_x}")
        print(f"line y1: {simulation.line_y1}")
        print(f"line y2: {simulation.line_y2}")
    elif simulation.source_type == 'point_source':
        print(f"source_x: {simulation.source_x}")
        print(f"source_y: {simulation.source_y}")
    print(f"c: {simulation.c}")
    print(f"epsilon_0: {simulation.epsilon_0}")
    print(f"mu_0: {simulation.mu_0}")
    print(f"geometries: {simulation.geometries}")
    print(f"sources: {simulation.sources}")
    print(f"detectors: {simulation.detectors}")
    