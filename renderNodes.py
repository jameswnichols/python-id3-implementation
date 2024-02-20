import pygame
from pygame import Vector2
from mainRewrite import *
import pickle

SCREEN_SIZE = Vector2(800, 600)

class Object:
    def __init__(self, position : Vector2, renderSurface : pygame.Surface):
        self.position = position
        self.renderSurface = renderSurface
        self.renderRect = renderSurface.get_rect()

    def isVisible(self, screenRect : pygame.Rect):
        self.renderRect.topleft = self.position
        return screenRect.colliderect(self.renderRect)

screen = pygame.display.set_mode(SCREEN_SIZE)
running = True

cameraPosition = Vector2(0, 0)
cameraRect = pygame.Rect(cameraPosition, SCREEN_SIZE)

nodes = None
with open("nodesOutput.data", "rb") as f:
    nodes = pickle.load(f)

blankSurface = pygame.Surface((50, 50))
blankSurface.fill((255,0,0))

objects = [Object(Vector2(0, 0), blankSurface)]

while running:
    screen.fill((255,255,255))

    for object in objects:
        if not object.isVisible(cameraRect):
            continue
        objectPosition = object.position - cameraPosition
        screen.blit(object.renderSurface, objectPosition)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    pygame.display.flip()

pygame.quit()