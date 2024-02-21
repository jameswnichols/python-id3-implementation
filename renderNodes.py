import pygame
from pygame import Vector2
from mainRewrite import *
import pickle

pygame.init()

FONT = [pygame.font.SysFont("arialrounded", x) for x in range(1, 100)]

SCREEN_SIZE = Vector2(800, 600)

DISTRIBUTION_WIDTH = 400
DISTRIBUTION_HEIGHT_GAP = 150

class Object:
    def __init__(self, position : Vector2, renderSurface : pygame.Surface, nodeClass : str):
        self.position = position
        self.renderSurface = renderSurface
        self.renderRect = renderSurface.get_rect()
        self.renderRect.topleft = self.position

        self.fontSurface = FONT[12].render(nodeClass, True, (0, 0, 255))
        self.fontSurfacePosition = self.renderRect.center - Vector2(self.fontSurface.get_rect().center)

    def isVisible(self, screenRect : pygame.Rect):
        self.renderRect.topleft = self.position
        return screenRect.colliderect(self.renderRect)

    def render(self, screen : pygame.Surface, screenRect : pygame.Rect):
        screenPosition = self.position - screenRect.topleft
        screen.blit(self.renderSurface, screenPosition)
        screen.blit(self.fontSurface, self.fontSurfacePosition-screenRect.topleft)
    
class Line:
    def __init__(self, startPosition : Vector2, endPosition : Vector2, text : str):
        self.startPosition = startPosition
        self.endPosition = endPosition

        self.fontSurface = FONT[12].render(text, True, (0, 0, 255))
        fontLinePosition = self.startPosition.lerp(self.endPosition, 0.5)
        self.fontSurfacePosition = fontLinePosition - Vector2(self.fontSurface.get_rect().center)

        self.renderRect = pygame.Rect(self.startPosition, (self.endPosition + Vector2(1, 1)) - self.startPosition)
    
    def isVisible(self, screenRect : pygame.Rect):
        self.renderRect.topleft = self.startPosition
        return screenRect.colliderect(self.renderRect)

    def render(self, screen : pygame.Surface, screenRect : pygame.Rect):
        screenStartPosition = self.startPosition - screenRect.topleft
        screenEndPosition = self.endPosition - screenRect.topleft
        pygame.draw.line(screen, (155,155,155), screenStartPosition, screenEndPosition,3)
        screen.blit(self.fontSurface, self.fontSurfacePosition - screenRect.topleft)

def generateObjectsFromNodes(nodes : dict[Node], rootNodeName : str):
    nodesToAdd = [{"node":nodes[rootNodeName],"position":Vector2(0, 0)}]
    objects = []
    lines = []
    while len(nodesToAdd) > 0 :
        poppedNode = nodesToAdd.pop(0)
        nodeToAdd, nodePosition = poppedNode["node"], poppedNode["position"]
        childNodeStartPosition = nodePosition - Vector2(DISTRIBUTION_WIDTH/2, 0)
        childDistanceBetween = DISTRIBUTION_WIDTH / (len(nodeToAdd.children) + 1)
        for i, (joiningValue, child) in enumerate(nodeToAdd.children.items()):
            childPosition = childNodeStartPosition + Vector2(childDistanceBetween * (i + 1), DISTRIBUTION_HEIGHT_GAP)
            lines.append(Line(nodePosition + Vector2(12.5, 12.5), childPosition + Vector2(12.5, 12.5),joiningValue))
            nodesToAdd.append({"node":child,"position":childPosition})
        
        objects.append(Object(nodePosition, blankSurface, nodeToAdd.paramClass))
    return lines, objects
        
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

blankSurface = pygame.Surface((25, 25))
blankSurface.fill((255,0,0))

lines, objects = generateObjectsFromNodes(nodes, "Root")#[Object(Vector2(0, 0), blankSurface)]

while running:
    screen.fill((255,255,255))

    cameraRect.topleft = cameraPosition

    for line in lines:
        if not line.isVisible(cameraRect):
            continue
        
        line.render(screen, cameraRect)

    for object in objects:
        if not object.isVisible(cameraRect):
            continue

        object.render(screen, cameraRect)

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