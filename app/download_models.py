"""Downloads the main model and the secondary model during setup. The others download from the app. First it looks
for models other apps on this PC already have, and uses those where they are instead of downloading them."""
import finder
import models


def show(message):
    print("\r" + message.ljust(70), end="", flush=True)


show("Looking for models you already have")
found = models.find_copies(finder.model_roots(finder.find_other_apps()), show)
if found:
    print("\rFound in other apps, so not downloaded: " + ", ".join(models.LABELS.get(n, n) for n in found), flush=True)
for name in (models.MAIN, "secondary"):
    models.ensure(name, show)
print("\rModels are ready.".ljust(71), flush=True)
