"""Downloads the main model and the secondary model during setup. The others download from the app."""
import models


def show(message):
    print("\r" + message.ljust(70), end="", flush=True)


for name in (models.MAIN, "secondary"):
    models.ensure(name, show)
print("\rModels are ready.".ljust(71), flush=True)
