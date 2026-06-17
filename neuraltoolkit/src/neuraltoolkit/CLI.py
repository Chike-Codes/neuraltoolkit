import sys

def progress_bar(frac, bar_length=40, front_str="", end_str=""):
    filled_length = int(bar_length * frac)

    # block character unicode escape: \u2588
    bar = "\u2588" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\r{front_str} |{bar}| {end_str}")
    sys.stdout.flush()

def epoch_summary(config, metrics, epoch):
    # accepts training config
    sys.stdout.write((f"\rEpoch: {epoch} / {config.epochs} "
                      f"- Loss: {metrics.loss:.5f} "))
    
    if metrics.val_loss != None:
        sys.stdout.write(f"- Validation Loss: {metrics.val_loss:.5f}")
    
    sys.stdout.write(" " * 100 + "\n")
