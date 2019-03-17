import pyglet

window = pyglet.window.Window(width=360, height=200)

@window.event

def on_key_press(key,modifiers):
    if (key == pyglet.window.key.UP):
        print("MOVE UP")
    elif (key == pyglet.window.key.DOWN):
        print("MOVE DOWN")
    elif (key == pyglet.window.key.LEFT):
        print("MOVE LEFT ")
    elif (key == pyglet.window.key.RIGHT):
        print("MOVE RIGHT")

pyglet.app.run()
