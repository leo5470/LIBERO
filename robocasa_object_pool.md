# RoboCasa object pool & graspability filter (RICL unseen-object experiment)

> Machine-readable reference for the RICL-with-unseen-objects experiment. It defines which
> RoboCasa365 object categories are valid **pick-and-place manipulanda** for LIBERO, which to
> exclude, and which to repurpose as place-targets. Lists are copy-paste ready. The canonical
> source of truth is RoboCasa's `OBJ_CATEGORIES`; §6 regenerates every list below from it, so
> if a hand-copied name ever drifts, trust the snippet output.

## 1. Filter rules (apply in this order)

1. **Drop `graspable == False`.** RoboCasa tags every category with a `graspable` flag in
   `OBJ_CATEGORIES`. `graspable=False` means their own pipeline never grasps it directly (it gets
   placed in/on a container), which maps almost exactly to "bad top-down pick-and-place target."
   **41 of 198** categories are non-graspable — exclude them as manipulanda.
2. **Repurpose receptacles as the place-target.** 7 of those 41 are `types=("receptacle")`
   (plate, tray, basket, cutting_board, oven_tray, baking_sheet, placemat). Keep them in the scene
   as the *destination* of pick-and-place, just never as the grasped object.
3. **Exclude LIBERO-overlap categories from the *unseen* pools.** To guarantee objects are unseen
   by your LIBERO π0-FAST base, drop categories that already exist in LIBERO (semantic match — LIBERO
   names instances like `akita_black_bowl`, RoboCasa uses generic `bowl`). 16 core overlaps; 3 of
   them (plate/tray/basket) are already non-graspable, so only **13 graspable overlaps** need
   dropping here. 9 more are *borderline* (loose matches to LIBERO's HOPE groceries) — exclude or
   keep per how strict you want "unseen" to be.
4. **Empirical yield gate (the real filter).** `graspable=True` is necessary, not sufficient: tall /
   fragile / handled / elongated objects (e.g. `baguette`, `wine_glass`, `kettle_non_electric`,
   `teapot`, `pitcher`, `jug`) often defeat a simple top-down grasp. Keep only objects where the
   scripted policy clears a success-yield threshold (e.g. ≥ 30–50%), and report the per-object
   scripted ceiling so RICL success is normalized against achievability.

Net result: ~**135** clean graspable + unseen categories (+9 borderline = 144), drawn from 157
graspable total, before the empirical gate.

## 2. Counts

| Set | Count |
|---|---|
| All categories in `OBJ_CATEGORIES` | 198 |
| `graspable=False` (exclude as manipulanda) | 41 |
| &nbsp;&nbsp;of which receptacles (reuse as place-target) | 7 |
| `graspable=True` | 157 |
| LIBERO-overlap, core, graspable (exclude from unseen pools) | 13 |
| LIBERO-overlap, borderline, graspable (optional exclude) | 9 |
| **Manipuland pool (graspable, not core-overlap)** | **144** |
| &nbsp;&nbsp;strict (also dropping borderline) | 135 |

## 3. Lists (copy-paste ready)

```python
# RoboCasa categories with graspable == False  -> NOT valid pick-and-place manipulanda
NON_GRASPABLE = [
    "bagel", "bagged_food", "chips", "chocolate", "cutting_board", "fork", "knife", "plate",
    "scissors", "spatula", "spoon", "tray", "waffle", "dates", "ham", "cabbage", "lettuce",
    "ice_cube_tray", "cantaloupe", "spaghetti_box", "bottle_opener", "baking_sheet", "bacon",
    "watermelon", "raspberry", "coconut", "cauliflower", "lollipop", "can_opener", "pineapple",
    "sandwich_bread", "digital_scale", "flour_bag", "oven_tray", "pickle_slice", "tomato_slice",
    "turkey_slice", "pancake", "pizza", "basket", "placemat",
]  # 41

# Subset of NON_GRASPABLE: keep in the scene as the pick-and-place DESTINATION, not grasped
RECEPTACLE_TARGETS = [
    "plate", "tray", "basket", "cutting_board", "oven_tray", "baking_sheet", "placemat",
]  # 7

# Categories that already exist in LIBERO (semantic) -> exclude from the *unseen* pools.
LIBERO_OVERLAP_CORE = [
    "bowl", "plate", "basket", "pan", "mug", "ketchup", "milk", "corn", "wine", "tray",
    "mayonnaise", "butter_stick", "cream_cheese_stick", "cherry", "colander", "strainer",
]  # 16 (plate/tray/basket also non-graspable)
LIBERO_OVERLAP_BORDERLINE = [
    "can", "canned_food", "boxed_food", "cup", "coffee_cup", "glass_cup",
    "condiment_bottle", "tupperware", "juice",
]  # 9 (loose matches to LIBERO HOPE groceries)

# Final candidate manipulanda = graspable AND not a core LIBERO overlap (borderline kept, tagged).
MANIPULAND_POOL = [
    # --- fruits & nuts ---
    "apple", "banana", "kiwi", "lemon", "lemon_wedge", "lime", "mango", "orange", "peach", "pear",
    "tangerine", "grapes", "strawberry", "pomegranate", "apricot", "walnut",
    # --- vegetables ---
    "avocado", "bell_pepper", "broccoli", "carrot", "cucumber", "eggplant", "garlic", "mushroom",
    "onion", "potato", "squash", "sweet_potato", "tomato", "ginger", "chili_pepper", "celery",
    "brussel_sprout", "asparagus", "zucchini", "beet", "radish", "artichoke", "tofu", "pickle",
    # --- meat, poultry & seafood ---
    "egg", "fish", "steak", "shrimp", "chicken_breast", "lobster", "lamb_chop", "pork_loin",
    "pork_chop", "sausage", "salami", "scallops", "chicken_drumstick",
    # --- bread & bakery ---
    "baguette", "bread", "bread_flat", "croissant", "cake", "cupcake", "donut", "scone", "hotdog_bun",
    # --- prepared foods ---
    "hot_dog", "burrito", "sushi", "kebabs", "hamburger", "tacos", "dumpling",
    # --- sweets & spices ---
    "candy", "ice_cream", "jello_cup", "sugar_cube", "cookie_dough_ball", "marshmallow",
    "cinnamon", "paprika", "turmeric",
    # --- drinks ---
    "beer", "bottled_drink", "bottled_water", "boxed_drink", "liquor", "lemonade", "water_bottle",
    "thermos", "straw",
    # --- condiments, oils & jars ---
    "jam", "peanut_butter", "syrup_bottle", "honey_bottle", "olive_oil_bottle", "canola_oil",
    "oil_and_vinegar_bottle", "vinegar", "salsa", "mustard",
    # --- cookware (watch the empirical gate: handles/tall) ---
    "pot", "saucepan", "saucepan_with_lid", "saucepan_lid", "kettle_non_electric", "teapot",
    # --- dishes & containers ---
    "jug", "jug_wide_opening", "pitcher", "jar", "measuring_cup", "wine_glass", "shaker",
    "salt_and_pepper_shaker", "blender_jug", "aluminum_foil", "ice_cube", "skewers", "kebab_skewer",
    # --- utensils ---
    "ladle", "whisk", "tongs", "wooden_spoon",
    # --- tools & gadgets ---
    "rolling_pin", "cheese_grater", "pizza_cutter", "reamer", "peeler",
    # --- cleaning & household ---
    "sponge", "soap_dispenser", "bar_soap", "spray", "dish_brush", "candle",
    # --- dairy ---
    "cheese", "yogurt",
    # --- other packaged ---
    "bar", "cereal",
]  # 135 strict

# Optionally add the 9 borderline categories to MANIPULAND_POOL if you allow loose-overlap objects:
# MANIPULAND_POOL += LIBERO_OVERLAP_BORDERLINE   # -> 144
```

## 4. Suggested split usage (ties into experiment plan §3)

- **Seen (priming)**, **Tier A (unseen instance)**, and **Tier B (unseen category)** are all drawn
  from `MANIPULAND_POOL` only. Never put a `NON_GRASPABLE` category in any manipuland tier.
- **Tier B (unseen *category*)** is richest from the **fruits / vegetables / meat / bakery / prepared**
  groups — LIBERO has almost no raw produce or proteins, so these are unambiguously novel to the base.
- **Tier A (unseen *instance*)** = hold out instances of seen categories; use RoboCasa's per-category
  instance list (`objaverse` / `aigen` ids) and split instance ids disjointly.
- Keep one or two `RECEPTACLE_TARGETS` fixed as the destination across *all* tiers, so only the grasped
  object varies.

## 5. Caveats

- `graspable` is RoboCasa metadata, not a guarantee for *our* top-down policy — §1 rule 4 (empirical
  gate) is the binding filter. Expect attrition among tall/handled cookware and fragile glassware.
- Some `graspable=True` items are still large-ish; check the `scale` field per category and the
  per-instance bbox.
- RoboCasa365's powered **appliances** (blender, toaster, stand mixer, dishwasher, oven, coffee
  machine, …) are **fixtures**, not in `OBJ_CATEGORIES`, so they are not in any list here.
- The `washable / cookable / microwavable / fridgable / freezable` flags are irrelevant to grasping;
  ignore them unless you later add non-pick-place skills.

## 6. Regenerate authoritatively (run inside the robocasa env)

```python
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

graspable     = sorted(k for k, v in OBJ_CATEGORIES.items() if v.get("graspable") is True)
non_graspable = sorted(k for k, v in OBJ_CATEGORIES.items() if v.get("graspable") is False)

def types_of(v):
    t = v.get("types", ())
    return {t} if isinstance(t, str) else set(t)

receptacles = sorted(k for k in non_graspable if "receptacle" in types_of(OBJ_CATEGORIES[k]))

print(len(OBJ_CATEGORIES), "categories;", len(graspable), "graspable;", len(non_graspable), "not")
print("receptacles (reuse as targets):", receptacles)

# group the graspable pool by RoboCasa's own type taxonomy
from collections import defaultdict
by_type = defaultdict(list)
for k in graspable:
    for t in (types_of(OBJ_CATEGORIES[k]) or {"untyped"}):
        by_type[t].append(k)
for t, ks in sorted(by_type.items()):
    print(t, len(ks), sorted(ks))
```

This prints the ground-truth `graspable` / `non_graspable` / `receptacles` sets and a type-grouped
view; reconcile against §3 and treat the script output as canonical.
