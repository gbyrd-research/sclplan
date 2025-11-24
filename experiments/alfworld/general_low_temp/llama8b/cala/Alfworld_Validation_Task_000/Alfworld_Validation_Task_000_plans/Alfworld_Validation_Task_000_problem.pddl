(define (problem temp_problem) (:domain alfred)
(:objects
	bed_1 - bed
	desk_1 desk_2 - desk
	drawer_1 drawer_2 drawer_3 drawer_4 drawer_5 drawer_6 - drawer
	garbagecan_1 - garbagecan
	laundryhamper_1 - laundryhamper
	safe_1 - safe
	shelf_1 shelf_2 shelf_3 shelf_4 shelf_5 shelf_6 - shelf
	desklamp_1 - desklamp
	keychain_1 keychain_2 keychain_3 - keychain
	mug_1 mug_2 - mug
	pen_1 pen_2 pen_3 - pen
	pencil_1 pencil_2 pencil_3 - pencil
	alarmclock_1 - alarmclock
	bowl_1 - bowl
	cd_1 cd_2 cd_3 - cd
	cellphone_1 cellphone_2 cellphone_3 - cellphone
	creditcard_1 creditcard_2 - creditcard
	book_1 book_2 - book
	laptop_1 laptop_2 - laptop
	pillow_1 pillow_2 - pillow
)
(:init
	(isReceptacle bed_1)
	(isReceptacle desk_1)
	(isReceptacle desk_2)
	(isReceptacle drawer_1)
	(openable drawer_1)
	(opened drawer_1)
	(isReceptacle drawer_2)
	(openable drawer_2)
	(opened drawer_2)
	(isReceptacle drawer_3)
	(atReceptacleLocation drawer_3)
	(openable drawer_3)
	(opened drawer_3)
	(isReceptacle drawer_4)
	(openable drawer_4)
	(opened drawer_4)
	(isReceptacle drawer_5)
	(openable drawer_5)
	(opened drawer_5)
	(isReceptacle drawer_6)
	(openable drawer_6)
	(opened drawer_6)
	(isReceptacle garbagecan_1)
	(isReceptacle laundryhamper_1)
	(isReceptacle safe_1)
	(openable safe_1)
	(opened safe_1)
	(isReceptacle shelf_1)
	(isReceptacle shelf_2)
	(isReceptacle shelf_3)
	(isReceptacle shelf_4)
	(isReceptacle shelf_5)
	(isReceptacle shelf_6)
	(inReceptacle desklamp_1 shelf_3)
	(isLight desklamp_1)
	(inReceptacle keychain_1 desk_1)
	(inReceptacle mug_1 desk_1)
	(inReceptacle pen_1 desk_1)
	(inReceptacle pen_2 shelf_3)
	(inReceptacle pencil_1 shelf_3)
	(inReceptacle alarmclock_1 desk_2)
	(inReceptacle bowl_1 desk_2)
	(inReceptacle cd_1 desk_2)
	(inReceptacle cellphone_1 shelf_1)
	(inReceptacle creditcard_1 shelf_1)
	(inReceptacle mug_2 shelf_2)
	(inReceptacle pencil_2 shelf_4)
	(inReceptacle cd_2 garbagecan_1)
	(inReceptacle book_1 bed_1)
	(inReceptacle cellphone_2 bed_1)
	(inReceptacle laptop_1 bed_1)
	(inReceptacle laptop_2 bed_1)
	(inReceptacle pillow_1 bed_1)
	(inReceptacle pillow_2 bed_1)
	(inReceptacle keychain_2 safe_1)
	(inReceptacle keychain_3 safe_1)
	(inReceptacle creditcard_2 drawer_1)
	(inReceptacle pencil_3 drawer_2)
	(inReceptacle cd_3 drawer_4)
	(inReceptacle pen_3 drawer_4)
	(inReceptacle book_2 drawer_5)
	(inReceptacle cellphone_3 drawer_3)
)
(:goal
(exists (?o - bowl ?l - desklamp)
(and (isLight ?l)
(inReceptacle ?o (under ?l))))
(examined ?o ?l)))