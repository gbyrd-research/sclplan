EXAMPLES = """
Example 1:
Task: You should place a fork in the drawer, an apple in the fridge, and turn on the faucet.
Discovered Objects: Fork_1, Bottle_1, Drawer_1, Drawer_2, Cabinet_1, Fridge_1, Vase_1, Faucet_1, Apple_1, Fork_2, Knife_1, Drawer_3
Goal State:(:goal
    (exists (?fo - fork ?l - drawer ?a - apple ?fr - fridge ?fa - faucet)
    (and 
        (objectInReceptacle ?fo ?d)
        (objectInReceptacle ?a ?fr)
        (isToggled ?fa)
)))

Done!

EXAMPLE 2:
Task: You should heat up some food.
Discovered Objects: Fork_1, Plate_1, Drawer_1, Microwave_1, Egg_1, Apple_1, Knife_1, Fridge_1, Fauce_1, Vase_1, WineBottle_1, Shelf_1
Goal State:(:goal
    (exists (?e - egg ?p - plate)
    (and
        (objectInReceptacle ?e ?p)
        (isCooked ?e)
)))

Done!

EXAMPLE 3:
Task: You should close the microwave.
Discovered Objects: Spoon_1, TrashCan_1, Cabinet_1, Sink_1, LightSwitch_1, Cabinet_2, Spoon_2, Microwave_1
Goal State: (:goal
    (exists (?m - microwave)
    (and
        (isClosed ?m)
)))

Done!
"""