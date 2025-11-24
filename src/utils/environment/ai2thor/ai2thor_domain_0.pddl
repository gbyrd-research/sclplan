(define (domain custom_ai2thor)
    (:requirements :strips :action-costs :disjunctive-preconditions)

    (:types
        agent 
        location
        object - location
        receptacle - object
    )

    (:predicates
        ; object attribute predicates
        (toggleable ?o - object) ; true if the object is toggleable
        (isToggled ?o - object) ; true if the object is isToggled
        (sliceable ?o - object) ; true if the object is sliceable
        (isSliced ?o - object) ; true if the object is isSliced
        (isKnife ?o - object)   ; true if the object is some form of knife
        (openable ?o - object) ; true if the object is openable
        (isOpen ?o - object) ; true if the object is isOpen
        (pickupable ?o - object) ; true if the object is pickupable
        (isPickedUp ?o - object) ; true if the object is isPickedUp

        ; agent state predicates
        (isHoldingObject ?a - agent) ; true if the agent is holding an object
        
        ; relational predicates
        (isValidReceptacle ?o - object ?r - receptacle) ; true if the receptacle is a valid receptacle for the object
        (objectInReceptacle ?o - object ?r - receptacle) ; true if the object is in the receptacle
        (objectHeld ?a - agent ?o - object) ; true if the agent is holding the object
        (atLocation ?a - agent ?l - location) ; true if the specified agent is at the location
    )

    (:action MoveAgentToLocation
        :parameters (?a - agent ?lStart - location ?lEnd - location)
        :precondition (and (not (atLocation ?a ?lEnd)); there is no agent at the target end location
                           (atLocation ?a ?lStart)      ; the target agent is at the specified starting location
                           )
        :effect (and (atLocation ?a ?lEnd)            ; an agent appears at the end location
                     (not (atLocation ?a ?lStart))    ; an agent is no longer at the starting location
                     (atLocation ?a ?lEnd)              ; the target agent is at the specified end location
                     (not (atLocation ?a ?lStart))      ; the target agent is no longer at the specified starting location
                     )
    )

    (:action OpenObject
        :parameters (?a - agent ?o - object)
        :precondition (and (atLocation ?a ?o)           ; the target agent must be at the target object
                           (openable ?o)                ; the target object must be openable
                           (not (isOpen ?o))            ; the object must not already be open
                           (not (isHoldingObject ?a))   ; the agent must not be already holding an object to open one
                           )
        :effect (and (isOpen ?o))                       ; the target object is opened
    )

    (:action CloseObject
        :parameters (?a - agent ?o - object)
        :precondition (and (atLocation ?a ?o)           ; the target agent must be at the target object
                           (openable ?o)                ; the target object must be openable
                           (isOpen ?o)                  ; the object must already be open
                           (not (isHoldingObject ?a))   ; the agent must not be already holding an object to open one
                           )
        :effect (and (not (isOpen ?o)))                 ; the target object is not opened
    )

    (:action PickUpObject
        :parameters (?a - agent ?o - object)
        :precondition (and (atLocation ?a ?o)           ; the target agent must be at the target object
                           (pickupable ?o)              ; the target object must be pickupable
                           (not (isPickedUp ?o))        ; the target object must not already be picked up
                           (not (isHoldingObject ?a))   ; the target agent must not already by holding an object
                           )
        :effect (and (isHoldingObject ?a)               ; the target agent is not holding an object
                     (isPickedUp ?o)                    ; the target object is picked up
                     (objectHeld ?a ?o)                 ; the target agent is now holding the target object
                     )
    )

    (:action PlaceObject
        :parameters (?a - agent ?heldObject - object ?targetReceptacle - receptacle)
        :precondition (and (isValidReceptacle ?heldObject ?targetReceptacle) ; the target receptacle must be a valid receptacle for the held object
                           (objectHeld ?a ?heldObject)  ; the target held object must be held by the target agent
                           (atLocation ?a ?targetReceptacle) ; the target agent must be at the target receptacle
                           (or (not (openable ?targetReceptacle)) ; either the target receptacle is not openable
                               (isOpen ?targetReceptacle))  ; or the target receptacle is already open
                           )
        :effect (and (not (objectHeld ?a ?heldObject))  ; the target agent is no longer holding the target held object
                     (not (isPickedUp ?heldObject))     ; the target held object is not longer picked up by an agent
                     (not (isHoldingObject ?a))         ; the target agent is no longer holding an object
                     (objectInReceptacle ?heldObject ?targetReceptacle) ; the target held object is not in the target receptacle
                     )
    )

    (:action ToggleObjectOn
        :parameters (?a - agent ?o - object)
        :precondition (and (toggleable ?o)              ; the target object must be togglable
                           (not (isToggled ?o))         ; the target object must be toggled off
                           (atLocation ?a ?o)           ; the target agent must be at the target object
                           (not (isHoldingObject ?a))   ; the target agent must not be holding something
                           )
        :effect (and (isToggled ?o)                     ; the target object is toggled on
                     )
    )

    (:action ToggleObjectOff
        :parameters (?a - agent ?o - object)
        :precondition (and (toggleable ?o)              ; the target object must be toggleable
                           (isToggled ?o)               ; the target object must be toggled on
                           (atLocation ?a ?o)           ; the target agent must be at the target object
                           (not (isHoldingObject ?a))   ; the target agent must not be holding something
        )
        :effect (and (not (isToggled ?o)))              ; the target object is toggled off
    )

    (:action SliceObject
        :parameters (?a - agent ?heldObject - object ?o - object)
        :precondition (and (isHoldingObject ?a)         ; the target agent must be holding an object
                           (isKnife ?heldObject)           ; the target held object must be a knife
                           (objectHeld ?a ?heldObject)     ; the target agent must actually be holding the target held object
                           (atLocation ?a ?o)           ; the target agent must be at the location of the object to be sliced
                           (sliceable ?o)               ; the target object to be sliced must be sliceable
                           (not (isSliced ?o))          ; the target object to be sliced must not already be sliced
                           )
        :effect (and (isSliced ?o))                     ; the target object to slice is now sliced
    )
)
