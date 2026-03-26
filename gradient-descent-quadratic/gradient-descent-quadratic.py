def objective_function(x, a, b, c):
    """Calculates f(x) = ax^2 + bx + c"""
    return a * (x**2) + (b * x) + c

def objective_function_derivative(x, a, b):
    """Calculates the derivative f'(x) = 2ax + b'"""
    return (2 * a * x) + b

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Set x with our initial x positionl
    x = x0
    for i in range(steps):
        # Step A: Find the slope at the current position
        gradient = objective_function_derivative(x, a, b)

        # Step B: Update x by moving in the OPPOSITE direction of the slope
        x = x - (lr * gradient)

        # Step C: Check our progress over the objective function
        f_x = objective_function(x, a, b, c)
        print(f"Iteration {i}: x = {x:.6f}, f(x) = {f_x:.6f}")

    print(f"Minimum found at x = {x:.6f} with steps = {steps}")
    return x