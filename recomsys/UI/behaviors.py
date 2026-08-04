from kivy.animation import Animation


class HoverBehavior:
    """
    Behavior پایه برای Hover
    """

    hovered = False

    def on_hover_enter(self):
        """
        Override
        """
        pass

    def on_hover_leave(self):
        """
        Override
        """
        pass


class PressBehavior:
    """
    Behavior پایه برای کلیک
    """

    pressed = False

    def on_press_animation(self):

        Animation.cancel_all(self)

        Animation(
            scale=0.97,
            d=0.08
        ).start(self)

    def on_release_animation(self):

        Animation.cancel_all(self)

        Animation(
            scale=1,
            d=0.12
        ).start(self)


class GlowBehavior:
    """
    Behavior مربوط به Glow
    """

    glow_alpha = 0.12

    def glow_in(self):

        Animation.cancel_all(self, "glow_alpha")

        Animation(
            glow_alpha=0.35,
            d=.20
        ).start(self)

    def glow_out(self):

        Animation.cancel_all(self, "glow_alpha")

        Animation(
            glow_alpha=0.12,
            d=.20
        ).start(self)


class ElevationBehavior:
    """
    افزایش Elevation هنگام Hover
    """

    normal_elevation = 3
    hover_elevation = 10

    def elevate(self):

        Animation.cancel_all(self)

        Animation(
            elevation=self.hover_elevation,
            d=.15
        ).start(self)

    def normalize(self):

        Animation.cancel_all(self)

        Animation(
            elevation=self.normal_elevation,
            d=.15
        ).start(self)


class FadeBehavior:
    """
    Fade In / Out
    """

    def fade_in(self):

        self.opacity = 0

        Animation(
            opacity=1,
            d=.35
        ).start(self)

    def fade_out(self):

        Animation(
            opacity=0,
            d=.35
        ).start(self)