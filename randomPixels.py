import random
import pygame

# Initialize Pygame
pygame.init()

# Set the window size and title
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Random Pixels')

# Run the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Clear the screen
    screen.fill((0, 0, 0))
    # Generate random x and y coordinates for the pixel
    x = random.randint(0, 640)
    y = random.randint(0, 480)
    # Draw the pixel at the random coordinates
    pygame.draw.rect(screen, (255, 255, 255), (x, y, 1, 1))
    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
