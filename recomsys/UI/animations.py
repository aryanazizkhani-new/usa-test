from kivy.animation import Animation


class UIAnimations:
    """
    تمام انیمیشن های پروژه
    """

    # ============================================
    # Fade
    # ============================================

    @staticmethod
    def fade_in(widget, duration=.30):

        widget.opacity = 0

        Animation(
            opacity=1,
            d=duration,
            t="out_quad"
        ).start(widget)

    @staticmethod
    def fade_out(widget, duration=.30):

        Animation(
            opacity=0,
            d=duration,
            t="out_quad"
        ).start(widget)

    # ============================================
    # Scale
    # ============================================

    @staticmethod
    def pop(widget):

        Animation.cancel_all(widget)

        (
            Animation(
                scale=1.08,
                d=.08
            ) +
            Animation(
                scale=1,
                d=.12
            )
        ).start(widget)

    @staticmethod
    def press(widget):

        Animation.cancel_all(widget)

        Animation(
            scale=.96,
            d=.08
        ).start(widget)

    @staticmethod
    def release(widget):

        Animation.cancel_all(widget)

        Animation(
            scale=1,
            d=.12,
            t="out_quad"
        ).start(widget)

    # ============================================
    # Elevation
    # ============================================

    @staticmethod
    def elevate(widget, value=10):

        Animation.cancel_all(widget)

        Animation(
            elevation=value,
            d=.15
        ).start(widget)

    @staticmethod
    def normalize(widget, value=3):

        Animation.cancel_all(widget)

        Animation(
            elevation=value,
            d=.18
        ).start(widget)

    # ============================================
    # Hover Move
    # ============================================

    @staticmethod
    def move(widget, x, y):

        Animation.cancel_all(widget)

        Animation(
            x=x,
            y=y,
            d=.08
        ).start(widget)

    @staticmethod
    def move_back(widget, x, y):

        Animation.cancel_all(widget)

        Animation(
            x=x,
            y=y,
            d=.20,
            t="out_quad"
        ).start(widget)

    # ============================================
    # Glow
    # ============================================

    @staticmethod
    def glow(widget):

        Animation.cancel_all(widget, "glow_alpha")

        widget.glow_alpha = .12

        anim = (

            Animation(
                glow_alpha=.35,
                d=.8,
                t="in_out_sine"
            )

            +

            Animation(
                glow_alpha=.12,
                d=.8,
                t="in_out_sine"
            )

        )

        anim.repeat = True

        anim.start(widget)

    @staticmethod
    def stop_glow(widget):

        Animation.cancel_all(widget, "glow_alpha")

        widget.glow_alpha = .12

    # ============================================
    # Shine
    # ============================================

    @staticmethod
    def shine(widget):

        Animation.cancel_all(widget.shine)

        start_x = widget.x - widget.width

        end_x = widget.x + widget.width

        widget.shine.pos = (
            start_x,
            widget.y
        )

        Animation(

            x=end_x,

            d=.8,

            t="out_quad"

        ).start(widget.shine)

    # ============================================
    # Intro
    # ============================================

    @staticmethod
    def intro(widget):

        widget.opacity = 0

        widget.y -= 30

        (

            Animation(
                opacity=1,
                y=widget.y + 30,
                d=.45,
                t="out_back"
            )

        ).start(widget)