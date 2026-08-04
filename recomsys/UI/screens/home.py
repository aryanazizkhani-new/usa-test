from kivy.metrics import dp
from kivy.animation import Animation
from kivy.clock import Clock

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel

from ui.gradient_background import GradientBackground
from ui.widgets.modern_card import ModernCard


class HomeScreen(MDScreen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bg = GradientBackground()
        self.add_widget(self.bg)

        self.root_box = MDBoxLayout(
            orientation="vertical",
            padding=dp(30),
            spacing=dp(25)
        )

        self.add_widget(self.root_box)

        self.cards = []

        self.build_header()
        self.build_cards()

        Clock.schedule_once(self.play_intro, .25)

    # -----------------------------------------------------

    def build_header(self):

        title = MDLabel(
            text="Decision AI",
            halign="center",
            font_style="H3",
            bold=True,
            size_hint_y=None,
            height=dp(60)
        )

        subtitle = MDLabel(
            text="Smart Recommendation Platform",
            halign="center",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(30)
        )

        self.root_box.add_widget(title)
        self.root_box.add_widget(subtitle)

    # -----------------------------------------------------

    def build_cards(self):

        rows = [

    [
        ("movie", "Movies", ""),
        ("book", "Books", ""),
    ],

    [
        ("music", "Musics", ""),
        ("application", "Apps", ""),
        ("robot", "Help", "")
    ],


]

        for row in rows:

            line = MDBoxLayout(

                orientation="horizontal",

                spacing=dp(20),

                adaptive_height=True,

                size_hint_y=None,

                height=dp(185),

            )

            for icon, title, subtitle in row:

                card = ModernCard(

                    icon=icon,

                    title=title,

                    subtitle=subtitle,

                )

                card.opacity = 0

                line.add_widget(card)

                self.cards.append(card)

            self.root_box.add_widget(line)

    # -----------------------------------------------------

    def play_intro(self, *_):

        delay = 0

        for card in self.cards:

            card.scale = .9

            Animation(

                opacity=1,

                scale=1,

                d=.45,

                t="out_back"

            ).start(card)

            delay += .08