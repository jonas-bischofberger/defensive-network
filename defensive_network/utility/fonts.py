import pathlib
import matplotlib.font_manager


FONTS_DIR = pathlib.Path(__file__).with_name("fonts")
print("Fonts directory:", FONTS_DIR)


def add_fonts(fonts_dir = FONTS_DIR, exts=(".ttf", ".otf", ".ttc", ".otc")) -> int:
    fonts_dir = pathlib.Path(fonts_dir)
    seen = set()  # avoid duplicates

    for p in fonts_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            key = p.resolve()
            if key in seen:
                continue
            try:
                matplotlib.font_manager.fontManager.addfont(str(p))
                print("Adding font:", p)
                seen.add(key)
            except Exception as e:
                print(f"Skipping {p} ({e})")

    # Optional: refresh Matplotlib’s cache of names (not strictly required on recent versions)
    matplotlib.font_manager._load_fontmanager(try_read_cache=False)

    # show installed fonts
    print("Available fonts:")
    for font in sorted(set(f.name for f in matplotlib.font_manager.fontManager.ttflist)):
        print(" -", font)


if __name__ == '__main__':
    add_fonts()
