EXAMPLES = """

Example 1: 

Task: Your task is to: put a clean plate in microwave.
Goal State: (:goal
    (exists (?t - plate ?r - microwave)
    (and 
        (inReceptacle ?t ?r)
        (isClean ?t)
)))

Done!

Example 2:

Task: Your task is to: examine an alarmclock with the desklamp.
Goal State: (:goal
    (exists (?t - alarmclock ?l - desklamp)
    (and 
        (examined ?t ?l) 
        (holds ?t)
)))

Done!

Example 3:

Task: Your task is to: put two cellphone in bed.
Goal State: (:goal
    (exists (?t1 - cellphone ?t2 - cellphone ?r - bed)
    (and 
        (inReceptacle ?t1 ?r)
        (inReceptacle ?t2 ?r)
        (not (= ?t1 ?t2))
)))

Done!

"""