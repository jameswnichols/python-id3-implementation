import pygame
from pygame import Vector2
from mainRewrite import *
import pickle

SCREEN_SIZE = Vector2(800, 600)

DISTRIBUTION_WIDTH = 600
DISTRIBUTION_HEIGHT_GAP = 150

class Object:
    def __init__(self, position : Vector2, renderSurface : pygame.Surface):
        self.position = position
        self.renderSurface = renderSurface
        self.renderRect = renderSurface.get_rect()

    def isVisible(self, screenRect : pygame.Rect):
        self.renderRect.topleft = self.position
        return screenRect.colliderect(self.renderRect)

def generateObjectsFromNodes(nodes : dict[Node], rootNodeName : str):
    nodesToAdd = [{"node":nodes[rootNodeName],"position":Vector2(0, 0)}]
    objects = []
    while len(nodesToAdd) > 0:
        poppedNode = nodesToAdd.pop(0)
        nodeToAdd, nodePosition = poppedNode["node"], poppedNode["position"]
        childNodeStartPosition = nodePosition - Vector2(DISTRIBUTION_WIDTH/2, 0)
        childDistanceBetween = DISTRIBUTION_WIDTH / (len(nodeToAdd.children) + 1)
        for i, child in enumerate(nodeToAdd.children.values()):
            childPosition = childNodeStartPosition + Vector2(childDistanceBetween * (i + 1), DISTRIBUTION_HEIGHT_GAP)
            nodesToAdd.append({"node":child,"position":childPosition})
        objects.append(Object(nodePosition, blankSurface))
    return objects
        
screen = pygame.display.set_mode(SCREEN_SIZE, vsync=True)
running = True

cameraPosition = Vector2(0, 0)
cameraRect = pygame.Rect(cameraPosition, SCREEN_SIZE)
cameraHeld = False
mouseHoldStartPosition = Vector2(0, 0)
cameraHoldStartPosition = Vector2(0, 0)

nodes = None
with open("nodesOutput.data", "rb") as f:
    nodes = pickle.load(f)

blankSurface = pygame.Surface((50, 50))
blankSurface.fill((255,0,0))

objects = generateObjectsFromNodes(nodes, "Root")#[Object(Vector2(0, 0), blankSurface)]

while running:
    screen.fill((255,255,255))

    cameraRect.topleft = cameraPosition

    for object in objects:
        if not object.isVisible(cameraRect):
            continue
        objectPosition = object.position - cameraPosition
        screen.blit(object.renderSurface, objectPosition)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if pygame.mouse.get_pressed()[0] and not cameraHeld:
        cameraHeld = True
        mouseHoldStartPosition = pygame.mouse.get_pos()
        cameraHoldStartPosition = cameraPosition
    elif pygame.mouse.get_pressed()[0] and cameraHeld:
        cameraPosition = cameraHoldStartPosition - (Vector2(pygame.mouse.get_pos())-mouseHoldStartPosition)
    elif not pygame.mouse.get_pressed()[0] and cameraHeld:
        cameraHeld = False

    pygame.display.flip()

pygame.quit()