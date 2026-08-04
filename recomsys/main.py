from kivymd.app import MDApp
from kivy.core.window import Window

from ui.screens.home import HomeScreen


class DecisionAI(MDApp):

    def build(self):
        # عنوان پنجره
        self.title = "Decision AI"

        # تم
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

        # حداقل اندازه پنجره (کامپیوتر)
        Window.minimum_width = 250
        Window.minimum_height = 50

        # اندازه اولیه پنجره
        Window.size = (1100, 700)

        # صفحه اصلی
        return HomeScreen()


if __name__ == "__main__":
    DecisionAI().run()