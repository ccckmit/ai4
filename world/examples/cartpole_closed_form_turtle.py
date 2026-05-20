"""
CartPole Closed-Form Controller with Turtle Graphics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple animation using Python's built-in turtle module.
"""

import world

def main():
    import turtle
    
    screen = turtle.Screen()
    screen.title("CartPole - Closed Form Controller")
    screen.setup(width=800, height=400)
    screen.tracer(0)
    
    # Cart
    cart = turtle.Turtle()
    cart.shape("square")
    cart.shapesize(stretch_wid=0.5, stretch_len=2)
    cart.color("blue")
    cart.penup()
    cart.goto(0, -50)
    
    # Pole
    pole = turtle.Turtle()
    pole.shape("arrow")
    pole.shapesize(stretch_wid=3, stretch_len=0.3)
    pole.color("red")
    pole.penup()
    pole.goto(0, 0)
    
    # Track
    track = turtle.Turtle()
    track.penup()
    track.goto(-350, -70)
    track.pendown()
    track.forward(700)
    
    # Info text
    info = turtle.Turtle()
    info.penup()
    info.goto(0, 150)
    info.hideturtle()
    
    env = world.make("CartPole-v1")
    observation, _ = env.reset(seed=42)
    steps = 0
    
    running = True
    def stop():
        nonlocal running
        running = False
    screen.onkeypress(stop, "q")
    screen.listen()
    
    while running:
        x, _, theta, _ = observation
        
        # Update cart position
        cart_x = x * 80
        cart.goto(cart_x, -50)
        
        # Update pole angle
        pole.setheading(-theta * 180 / 3.14159)
        pole.goto(cart_x, -30)
        
        info.clear()
        info.write(f"Steps: {steps}  x: {x:.2f}  θ: {theta*180/3.14159:.1f}°", 
                   align="center", font=("Arial", 12, "normal"))
        
        screen.update()
        
        # Control logic
        if observation[2] > 0:
            if observation[3] > 0.01:
                action = 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
            else:
                action = 0
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
        elif observation[2] < 0:
            if observation[3] < -0.01:
                action = 0
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
            else:
                action = 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
        
        if terminated or truncated:
            print(f"Episode ended: {steps} steps")
            steps = 0
            observation, _ = env.reset()
    
    screen.bye()

if __name__ == "__main__":
    main()