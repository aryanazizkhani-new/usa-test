from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivymd.uix.floatlayout import MDFloatLayout


class GradientBackground(MDFloatLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.colors = [
            (0.03, 0.05, 0.15, 1),  # dark blue
            (0.05, 0.35, 0.85, 1),  # blue
            (0.35, 0.15, 0.75, 1),  # purple
            (0.1, 0.75, 0.55, 1),   # green
            (1.0, 0.55, 0.1, 1),    # orange
            (0.9, 0.15, 0.15, 1),   # red
            (1.0, 0.85, 0.15, 1),   # yellow
            (0.45, 0.1, 0.75, 1),   # violet
            (0.1, 0.6, 1.0, 1),     # light blue
            (0.03, 0.05, 0.15, 1)
        ]


        self.index = 0
        self.progress = 0


        with self.canvas.before:

            self.color = Color()

            self.rect = Rectangle()


        self.bind(
            pos=self.update_canvas,
            size=self.update_canvas
        )


        Clock.schedule_interval(
            self.animate_gradient,
            1 / 60
        )


    # ---------------------------------------------

    def update_canvas(self, *args):

        self.rect.pos = self.pos
        self.rect.size = self.size


    # ---------------------------------------------

    def lerp(self, a, b, t):

        return a + (b - a) * t


    # ---------------------------------------------

    def animate_gradient(self, dt):

        current = self.colors[self.index]

        nxt = self.colors[
            (self.index + 1) % len(self.colors)
        ]


        self.progress += 0.002


        if self.progress >= 1:

            self.progress = 0

            self.index += 1

            if self.index >= len(self.colors)-1:
                self.index = 0


        t = self.progress


        new_color = (
            self.lerp(current[0], nxt[0], t),
            self.lerp(current[1], nxt[1], t),
            self.lerp(current[2], nxt[2], t),
            1
        )


        self.color.rgba = new_color