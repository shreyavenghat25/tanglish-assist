"""Hand-written Tanglish templates for synthetic training data.

Slots: {order} {item} {date} {amount}. A template only gets the entities it actually mentions,
so labels are correct by construction. Romanisation is deliberately inconsistent
(irukku/iruku, eppo/epo) because real users are inconsistent.

These templates produce TRAINING data only. The test set (data/test_seed.jsonl) is written
separately, in different words, so evaluation measures generalisation, not memorisation.
"""

ITEMS = [
    "shoe", "shoes", "phone cover", "mixie", "earphones", "earbuds", "kurta", "saree",
    "watch", "smart watch", "laptop bag", "charger", "power bank", "t-shirt", "jeans",
    "pressure cooker", "water bottle", "bedsheet", "headphones", "mobile", "tiffin box",
    "backpack", "trimmer", "kadai", "school bag", "lehenga", "speaker", "mouse", "keyboard",
    "tablet", "hair dryer", "induction stove", "dosa tawa", "silk saree", "kids dress",
]

DATES = [
    "nethu", "nethiku", "munnadi naal", "2 days", "3 days", "5 days", "one week", "10 days",
    "last week", "Monday", "saturday", "12th", "15th", "25th", "yesterday", "today morning",
    "inniku", "rendu naal", "moonu naal", "oru vaaram", "last month",
]

AMOUNTS = [199, 249, 299, 349, 399, 499, 549, 599, 799, 899, 999, 1199, 1299, 1499, 1799,
           1999, 2299, 2499, 2999, 3499, 4999, 6499, 8999, 12999, 15999]

PREFIXES = ["", "", "", "hi ", "hello ", "sir ", "anna ", "bro ", "vanakkam, ", "hello sir, ",
            "hi team, ", "madam ", "excuse me, "]
SUFFIXES = ["", "", "", " bro", " sir", " pls", " please", " anna", " asap", " 🙏", " 😡",
            " seekiram sollunga", " reply pannunga", " ??", "!!", " madam"]

TEMPLATES = {
    "order_status": [
        "en order {order} enga irukku nu sollunga",
        "{order} status enna?",
        "order {order} ippo enga iruku",
        "naan order panna {item} eppo varum",
        "{item} order pannirukken, track panna mudiyala, status sollunga",
        "en {item} ship aagiducha illaya",
        "order {order} out for delivery aa?",
        "{order} dispatch aagiducha nu check panni sollunga",
        "{item} enga varaikum vanthurukku",
        "order panna {item} oda current status enna",
        "tracking link work aagala, {order} enga irukku",
        "en parcel enga irukku nu theriyala, order id {order}",
        "{item} inniku delivery aaguma?",
        "order {order} confirm aagiducha?",
    ],
    "delivery_delay": [
        "promised date {date}, innum delivery aagala",
        "{order} {date} varum nu sonninga, innum varala",
        "{item} order panni {date} aachu, innum varala",
        "romba late aaguthu, {order} ku enna aachu",
        "delivery date {date}, ippo varaikum parcel varala",
        "already {date} delay, {item} eppo tharuveenga",
        "{date} ae deliver aaganum, ippo late aaguthu, very bad service",
        "delivery boy varave illa, {order} still pending",
        "ethana naal wait pannanum? {item} {date} la vanthirukkanum",
        "{order} delivery postpone aagitte irukku, enna problem",
        "{date} wait panren, {item} innum kaila varala",
        "expected delivery {date} nu potrukku, but no update",
    ],
    "cancel_order": [
        "intha order cancel pannanum",
        "{order} cancel pannunga",
        "{item} venam, order cancel panna mudiyuma",
        "wrong size select pannitten, {order} cancel pannunga",
        "{item} order mistake ah pannitten, cancel pls",
        "cancel button varala, {order} cancel pannunga",
        "{order} ship aagurathukku munnadi cancel pannidunga",
        "same {item} vera kadaila cheap ah kidaikuthu, cancel pannanum",
        "naan order cancel panna try panren, error varuthu, {order}",
        "{item} ippo thevai illa, cancel panni refund podunga",
        "double order aagiduchu, oru {item} cancel pannunga",
    ],
    "return_request": [
        "{item} size sariya varala, return pannalama?",
        "{order} return pannanum, fit aagala",
        "{item} pidikala, return eppadi pannurathu",
        "return window {date} thaan aachu, {item} return panna mudiyuma",
        "{item} colour photo la irundha maadhiri illa, return pannanum",
        "{order} la vandha {item} return pick up arrange pannunga",
        "{item} quality sari illa, return panren",
        "return request podanum, {order}",
        "{item} enakku small ah irukku, exchange illana return pannanum",
        "return option kaatala, {item} {date} munnadi thaan vanthuchu",
        "{item} use pannala, tag ellam irukku, return panna mudiyuma",
    ],
    "wrong_item": [
        "blue order panninen, red vanthuchu",
        "{order} la wrong item anupitteenga",
        "naan {item} order panninen, vera product vanthurukku",
        "{item} model vera, naan order pannathu idhu illa",
        "box la {item} illa, vera ethoo irukku",
        "size M order panninen, XL anupirukeenga, {order}",
        "{item} ku badhila empty box vanthuchu",
        "wrong {item} deliver aagirukku, replace pannunga",
        "{order} parcel la en order illa, vera aaloda item",
        "{item} brand vera, order page la vera brand potrukku",
        "two {item} order panninen, onnu thaan vanthuchu",
    ],
    "damaged_item": [
        "{item} box open panna crack aagi irukku",
        "{order} la vandha {item} damage aagi irukku",
        "{item} work aagala, first day ae problem",
        "{item} broken ah vanthurukku, replace pannunga",
        "parcel full ah kizhinju vanthuchu, {item} scratch irukku",
        "{item} on aagave illa, defective piece",
        "{item} leak aaguthu, {date} thaan vanthuchu",
        "{order} {item} screen la line varuthu",
        "{item} switch on panna smoke varuthu, romba bayama irukku",
        "glass {item} udanju irukku, packing sari illa",
        "{item} sound varala, defective ah irukku",
    ],
    "refund_status": [
        "refund eppo varum? {date} aachu",
        "{order} refund innum credit aagala",
        "return panni {date} aachu, money innum varala",
        "{amount} refund pending la irukku, eppo varum",
        "refund initiated nu message vanthuchu, bank la varala",
        "{order} ku refund {amount} eppo account la varum",
        "cancel panni {date} aachu, refund status enna",
        "refund original payment ku varuma illa wallet ku varuma, {order}",
        "{amount} rupees refund innum varala, {date} wait panren",
        "{item} return pick up aagiduchu, refund eppo",
        "refund failed nu kaatuthu, {amount} enna aachu",
    ],
    "payment_issue": [
        "UPI la {amount} cut aachu but order place aagala",
        "payment failed nu vanthuchu, aana {amount} debit aagiduchu",
        "{amount} rendu thadava cut aagiduchu",
        "card payment {date} pannen, order confirm aagala, money poiduchu",
        "COD option kaatala, payment panna mudiyala",
        "{order} ku double payment aagiduchu, {amount} extra pochu",
        "GPay la pay pannen, pending nu kaatuthu {date} la irundhu",
        "EMI select panna error varuthu",
        "{amount} debit aachu, order history la onnum illa",
        "net banking payment stuck aagiduchu, {amount} cut",
        "wallet balance {amount} irundhuchu, ippo zero kaatuthu",
    ],
    "address_change": [
        "delivery address maathanum, new flat ku anupunga",
        "{order} address change panna mudiyuma",
        "address la pin code thappa pottutten, {order}",
        "{item} office address ku deliver pannunga, veetu address venam",
        "phone number maathanum {order} la, delivery boy call panna mudiyathu",
        "naan oor ku poren, {order} delivery address change pannunga",
        "door number missing in address, {order} ku add pannunga",
        "{item} amma veetu ku anupa mudiyuma, address update pannanum",
        "landmark sari illa, delivery boy ku vazhi theriyala, {order}",
        "shipping address edit option illa, help pannunga",
    ],
    "coupon_issue": [
        "cashback {amount} promise pannanga, innum varala",
        "coupon code apply aagala",
        "{order} ku offer price kidaikala, {amount} extra charge pannirukeenga",
        "bank offer {amount} off nu potrundhuchu, apply aagala",
        "first order coupon work aagala",
        "{amount} cashback {date} la varum nu sonninga, varala",
        "coupon expired nu kaatuthu, aana inniku thaan vanthuchu",
        "{item} ku sale price kaatuchu, checkout la full price",
        "supercoins credit aagala, {order}",
        "referral bonus {amount} innum varala",
    ],
    "account_issue": [
        "OTP ye varala, login panna mudiyala",
        "account block aagiduchu, enna panrathu",
        "password reset link work aagala",
        "email id maathanum account la",
        "login panna 'something went wrong' nu varuthu",
        "en account la vera yaro order pannirukaanga, {order}",
        "pudhu number ku account transfer pannanum",
        "account delete pannanum, eppadi",
        "OTP late ah varuthu, expire aagiduthu",
        "{date} la irundhu app la login aagala",
        "profile la name thappa irukku, maatha mudiyala",
    ],
    "other": [
        "thanks bro, problem solve aagiduchu",
        "romba nandri, quick ah help panneenga",
        "hi",
        "vanakkam",
        "unga service super, keep it up",
        "customer care number enna",
        "ok sari, wait panren",
        "nalla experience, thank you team",
        "neenga bot ah illa human ah?",
        "store timing enna",
        "next sale eppo varum",
        "ok bye",
    ],
}

# Replies. {order_ref} expands to " ({order})" when an order ID is known, else "".
REPLIES = {
    "order_status": [
        "Kandippa check panren{order_ref}. Unga order ippo transit la irukku, tracking update SMS la varum.",
        "Sure! Unga order status check panni{order_ref} detail ah SMS la anupuren, konjam wait pannunga.",
    ],
    "delivery_delay": [
        "Delay ku sorry{order_ref}. Courier partner kitta escalate pannirukken, 24 hours la update tharuvom.",
        "Romba sorry, late aanathu engaloda thappu{order_ref}. Priority delivery ku mark pannirukken.",
    ],
    "cancel_order": [
        "Sari, unga order cancel request podren{order_ref}. Ship aagalana free ah cancel aagidum.",
        "Cancel request register pannitten{order_ref}. Prepaid na refund 5-7 working days la varum.",
    ],
    "return_request": [
        "Return pannalam{order_ref}! Return window kulla irundha pick-up schedule panren, tag remove pannatheenga.",
        "Sure, return request create pannitten{order_ref}. Pick-up date SMS la varum.",
    ],
    "wrong_item": [
        "Wrong item vanthathu ku romba sorry{order_ref}. Replacement arrange panren, wrong item pick-up pannuvom.",
        "Sorry for the mix-up{order_ref}! Photo upload pannunga, correct item free ah anupuvom.",
    ],
    "damaged_item": [
        "Damage aagi vanthathu ku sorry{order_ref}. Photo/video upload pannunga, replacement illana refund pannuvom.",
        "Romba sorry{order_ref}. Item use pannatheenga, replacement request create pannirukken.",
    ],
    "refund_status": [
        "Unga refund check panren{order_ref}. Initiate aagiyirundha bank ku 5-7 working days aagum.",
        "Refund status verify pannitten{order_ref}, bank side la process aaguthu. Innum delay aana sollunga, escalate panren.",
    ],
    "payment_issue": [
        "Kavalai padatheenga{order_ref}. Order fail aanaa debit aana amount 5-7 days la auto-refund aagidum.",
        "Payment issue ku sorry{order_ref}. Transaction ID share pannunga, payments team kitta check panren.",
    ],
    "address_change": [
        "Ship aagalana address change panna mudiyum{order_ref}. Pudhu address full ah pin code oda sollunga.",
        "Sure{order_ref}! Correct address/number sollunga, update panni confirm panren.",
    ],
    "coupon_issue": [
        "Offer apply aagathathu ku sorry{order_ref}. Details check panni eligible na amount credit pannuvom.",
        "Coupon terms check panren{order_ref}. Eligible na cashback 48 hours la credit aagum.",
    ],
    "account_issue": [
        "Account issue ku sorry. Registered mobile number sollunga, verify panni help panren.",
        "Kavalai vendam, app update panni try pannunga; innum problem na registered email sollunga, fix panren.",
    ],
    "other": [
        "Nandri! Vera ethavathu help venumna sollunga 😊",
        "Vanakkam! Unga order pathi ethavathu help venuma?",
    ],
}
