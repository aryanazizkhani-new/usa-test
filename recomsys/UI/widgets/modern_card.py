from kivy.animation import Animation
from kivy.core.window import Window
from kivy.metrics import dp
import os
from kivy.clock import Clock
from kivymd.uix.label import MDIcon
from kivy.core.window import Window
from kivy.graphics import (
    Color,
    RoundedRectangle,
    Rectangle,
    Line
)

from kivymd.uix.card import MDCard
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
FONT_PATH = os.path.join(
    os.path.dirname(__file__),
    "Segoe UI Emoji.ttf"
)

class ModernCard(MDCard):


    # --------------------------------------------------
    # Init
    # --------------------------------------------------

    def __init__(
        self,
        title="",
        subtitle="",
        icon="",
        **kwargs
    ):

        super().__init__(**kwargs)

    # -----------------------------
    # Text
    # -----------------------------

        self.title_text = title
        self.subtitle_text = subtitle
        self.icon_text = icon

    # -----------------------------
    # Hover
    # -----------------------------

        self.is_hover = False

    # -----------------------------
    # Style
    # -----------------------------

        self.setup_style()

    # -----------------------------
    # Canvas
    # -----------------------------

        self.setup_canvas()

    # -----------------------------
    # Layout
    # -----------------------------

        self.build_layout()

    # -----------------------------
    # Responsive
    # -----------------------------

        self.bind(
        size=self.update_responsive
    )
        Window.bind(size=self.update_columns)

        Clock.schedule_once(
        self.update_responsive,
        0
    )

    # -----------------------------
    # Mouse
    # -----------------------------

        Window.bind(
        mouse_pos=self.on_mouse_move
    )


    def update_responsive(self, *args):

        w = self.width


        # ------------------------------
        # Ultra Small
        # ------------------------------

        if w < dp(110):

            self.height = dp(140)

            self.padding = dp(8)

            self.spacing = dp(4)


            self.icon_label.font_size = dp(30)

            self.icon_label.height = dp(40)


            self.title_label.font_style = "H6"

            self.title_label.height = dp(30)


            self.subtitle_label.font_size = "9sp"



        # ------------------------------
        # Small Mobile
        # ------------------------------

        elif w < dp(150):

            self.height = dp(160)

            self.padding = dp(10)

            self.spacing = dp(6)


            self.icon_label.font_size = dp(38)

            self.icon_label.height = dp(55)


            self.title_label.font_style = "H6"

            self.title_label.height = dp(35)


            self.subtitle_label.font_size = "10sp"



        # ------------------------------
        # Normal Mobile
        # ------------------------------

        elif w < dp(220):

            self.height = dp(180)

            self.padding = dp(15)

            self.spacing = dp(10)


            self.icon_label.font_size = dp(50)

            self.icon_label.height = dp(75)


            self.title_label.font_style = "H5"

            self.title_label.height = dp(40)


            self.subtitle_label.font_size = "12sp"



        # ------------------------------
        # Tablet
        # ------------------------------

        elif w < dp(320):

            self.height = dp(210)

            self.padding = dp(20)

            self.spacing = dp(14)


            self.icon_label.font_size = dp(65)

            self.icon_label.height = dp(100)


            self.title_label.font_style = "H4"

            self.title_label.height = dp(45)


            self.subtitle_label.font_size = "14sp"



        # ------------------------------
        # Desktop
        # ------------------------------

        else:

            self.height = dp(190)

            self.padding = dp(18)

            self.spacing = dp(10)


            self.icon_label.font_size = dp(55)

            self.icon_label.height = dp(80)


            self.title_label.font_style = "H5"

            self.title_label.height = dp(40)


            self.subtitle_label.font_size = "13sp"
    def update_columns(self, window, size):

        width = size[0]


        if width < dp(400):

            self.cols = 1


        elif width < dp(700):

            self.cols = 2


        elif width < dp(1000):

            self.cols = 3


        else:

            self.cols = 4
            # --------------------------------------------------
    # Style
    # --------------------------------------------------

    def setup_style(self):


        self.orientation = "vertical"


        self.size_hint = (
    1,
    None
)

        self.height = dp(185)


        self.radius = [
            22
        ]


        self.elevation = 3


        self.padding = dp(20)


        self.spacing = dp(12)



        # -----------------------------
        # Colors
        # -----------------------------

        self.normal_bg = (

            0.08,
            0.11,
            0.18,
            0.92

        )


        self.hover_bg = (

            0.13,
            0.17,
            0.28,
            0.96

        )


        self.md_bg_color = self.normal_bg



        # -----------------------------
        # Hover Animation
        # -----------------------------

        self.normal_elevation = 3

        self.hover_elevation = 10
            # --------------------------------------------------
    # Canvas
    # --------------------------------------------------

    def setup_canvas(self):

        with self.canvas.before:


            # -----------------------------
            # Glass Background
            # -----------------------------

            Color(
                1,
                1,
                1,
                0.06
            )


            self.bg = RoundedRectangle(

                pos=self.pos,

                size=self.size,

                radius=[self.width * 0.08]

            )


            # -----------------------------
            # Border
            # -----------------------------

            Color(

                1,
                1,
                1,
                0.12

            )


            self.border = Line(

                rounded_rectangle=(

                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    22

                ),

                width=1.2

            )


            # -----------------------------
            # Shine
            # -----------------------------




        self.bind(

            pos=self.update_canvas,

            size=self.update_canvas

        )



    # --------------------------------------------------
    # Canvas Update
    # --------------------------------------------------

    def update_canvas(
        self,
        *args
    ):


        self.bg.pos = self.pos

        self.bg.size = self.size







        self.border.rounded_rectangle = (

    self.x,
    self.y,
    self.width,
    self.height,
    self.width * 0.08

)
            # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    def build_layout(self):


        self.content = MDBoxLayout(
    orientation="vertical",
    spacing=dp(15),
    padding=(0, dp(10)),
)

        self.icon_label = MDIcon(
    icon=self.icon_text,
    theme_text_color="Custom",
    text_color=(0.3, 0.8, 1, 1),
    font_size=dp(60),
    size_hint=(1, None),
    height=dp(90),
    halign="center",
    valign="middle",
)
        self.icon_label.text_size = (
    self.icon_label.width,
    self.icon_label.height
)

        self.title_label = MDLabel(

    text=self.title_text,

    halign="center",

    bold=True,

    font_style="H5",

    size_hint_y=None,

    height=dp(40)

)


        self.subtitle_label = MDLabel(

            text=self.subtitle_text,

            halign="center",

            theme_text_color="Secondary"

        )


        self.content.add_widget(
            self.icon_label
        )


        self.content.add_widget(
            self.title_label
        )


        self.content.add_widget(
            self.subtitle_label
        )


        self.add_widget(
            self.content
        )
            # --------------------------------------------------
    # Mouse Hover
    # --------------------------------------------------

    def on_mouse_move(
        self,
        window,
        pos
    ):

        if not self.get_root_window():

            return


        x, y = self.to_widget(
            *pos
        )


        if self.collide_point(
            x,
            y
        ):


            if not self.is_hover:

                self.enter_hover()



        else:


            if self.is_hover:

                self.leave_hover()



    # --------------------------------------------------
    # Enter Hover
    # --------------------------------------------------

    def enter_hover(self):

        self.is_hover = True


        Animation.cancel_all(
            self
        )


        self.hover_color()


        Animation(

            elevation=self.hover_elevation,

            scale=1.03,

            d=0.15,

            t="out_quad"

        ).start(self)



    # --------------------------------------------------
    # Leave Hover
    # --------------------------------------------------

    def leave_hover(self):

        self.is_hover = False


        Animation.cancel_all(
            self
        )


        self.normal_color()


        Animation(

            elevation=self.normal_elevation,

            scale=1,

            d=0.15,

            t="out_quad"

        ).start(self)
            # --------------------------------------------------
    # Hover Color
    # --------------------------------------------------

    def hover_color(self):

        self.md_bg_color = (

            0.13,
            0.17,
            0.28,
            0.96

        )



    # --------------------------------------------------
    # Normal Color
    # --------------------------------------------------

    def normal_color(self):

        self.md_bg_color = (

            0.08,
            0.11,
            0.18,
            0.92

        )



    # --------------------------------------------------
    # Glow
    # --------------------------------------------------

    def update_glow(self):

        if not hasattr(
            self,
            "border"
        ):

            return


        self.border.width = (

            1.2 +

            self.glow_alpha * 3

        )


        self.border.rounded_rectangle = (

            self.x,

            self.y,

            self.width,

            self.height,

            22

        )
            # --------------------------------------------------
    # Intro Animation
    # --------------------------------------------------

    def play_intro(self):

        self.opacity = 0

        Animation(

            opacity=1,

            d=.45,

            t="out_back"

        ).start(self)



    # --------------------------------------------------
    # Shine Animation
    # --------------------------------------------------

    def play_shine(self):
        pass
            # --------------------------------------------------
    # Enable Hover
    # --------------------------------------------------

    def enable_hover(self):

        self.is_hover = False



    # --------------------------------------------------
    # Disable Hover
    # --------------------------------------------------

    def disable_hover(self):

        self.is_hover = False



    # --------------------------------------------------
    # Stop Animation
    # --------------------------------------------------

    def stop_card_animation(self):

        Animation.cancel_all(
            self
        )



    # --------------------------------------------------
    # Remove Events
    # --------------------------------------------------

    def on_parent(
        self,
        instance,
        parent
    ):

        if parent is None:

            try:

                Window.unbind(
                    mouse_pos=self.on_mouse_move
                )

            except Exception:

                pass



    # --------------------------------------------------
    # Touch Shine Trigger
    # --------------------------------------------------

    def on_touch_down(
        self,
        touch
    ):

        if self.collide_point(
            *touch.pos
        ):

            self.play_shine()


        return super().on_touch_down(
            touch
        )