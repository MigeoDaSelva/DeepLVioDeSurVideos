import matplotlib.pyplot as plt
from configs import settings


class FigureRespository:
    @classmethod
    def save(self, file_name: str) -> None:
        plt.savefig(
            fname=f"{settings.IMAGES_PATH}/{file_name}.png",
            dpi=60,
            format="png",
        )
        plt.close()
        plt.cla()
        plt.clf()

    @classmethod
    def load(
        self,
        file_name: str,
    ) -> bytes:
        with open(f"{settings.IMAGES_PATH}/{file_name}.png", "rb") as file:
            image = file.read()
        return image
