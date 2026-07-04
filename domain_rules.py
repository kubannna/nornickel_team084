def classify_ore(talc_percent: float, normal_percent: float, fine_percent: float) -> dict:
    if talc_percent > 10:
        ore_class = "оталькованная"
        description = f"Руда классифицирована как оталькованная: содержание талька — {talc_percent:.1f}%"
    elif fine_percent > normal_percent:
        ore_class = "труднообогатимая"
        description = f"Руда классифицирована как труднообогатимая: преобладание тонких срастаний — {fine_percent:.1f}%"
    else:
        ore_class = "рядовая"
        description = f"Руда классифицирована как рядовая: преобладание обычных срастаний — {normal_percent:.1f}%"

    return {
        "class": ore_class,
        "talc_percent": talc_percent,
        "normal_percent": normal_percent,
        "fine_percent": fine_percent,
        "description": description
    }


if __name__ == "__main__":
    print(classify_ore(14, 20, 62))
    print(classify_ore(5, 30, 60))
    print(classify_ore(5, 70, 20))