const tripData = {
    fri_sep_4: {
    title: "Friday, Sept 4 - Troy to Wisconsin Dells",
    overnight: "Wisconsin Dells",
    summary: "Evening departure after work. Pure drive day. One combined dinner, fuel, and bathroom stop.",
    stops: [
      {
        name: "Troy",
        coords: [42.6064, -83.1498],
        note: "Leave around 5:00-6:00 PM Eastern."
      },
      {
        name: "Meal / Fuel Stop",
        coords: [41.8781, -87.6298],
        note: "Approximate Chicago-area stop. Keep it efficient."
      },
      {
        name: "Wisconsin Dells",
        coords: [43.6275, -89.7709],
        note: "Overnight. Expected late arrival around 1:00-2:00 AM Central."
      }
    ],
    segments: [
      {
        name: "Troy to Meal / Fuel Stop",
        from: [42.6064, -83.1498],
        to: [41.8781, -87.6298],
        time: "About 4-5 hr",
        buffer: "30-60 min",
        do: "Leave after work. Combine dinner, bathroom, and fuel into one stop.",
        skip: "No sightseeing. No long dinner.",
        detail: "This is just the first push west. Driver fatigue beats the schedule."
      },
      {
        name: "Meal / Fuel Stop to Wisconsin Dells",
        from: [41.8781, -87.6298],
        to: [43.6275, -89.7709],
        time: "About 3-4 hr",
        buffer: "30 min",
        do: "Drive directly to Wisconsin Dells.",
        skip: "No extra stops unless needed for safety.",
        detail: "Arrive late, check in, sleep."
      }
    ]
  },

  sat_sep_5: {
    title: "Saturday, Sept 5 - Wisconsin Dells to Bismarck",
    overnight: "Bismarck",
    summary: "Pure transportation day. Two driver changes, one proper meal stop, fuel and bathroom stops only.",
    stops: [
      {
        name: "Wisconsin Dells",
        coords: [43.6275, -89.7709],
        note: "Leave around 9:00 AM Central."
      },
      {
        name: "Minneapolis Area",
        coords: [44.9778, -93.2650],
        note: "Good region for meal, fuel, and driver switch."
      },
      {
        name: "Fargo Area",
        coords: [46.8772, -96.7898],
        note: "Good later fuel / stretch / driver switch area."
      },
      {
        name: "Bismarck",
        coords: [46.8083, -100.7837],
        note: "Overnight. Expected arrival around 8:00-9:00 PM Central."
      }
    ],
    segments: [
      {
        name: "Wisconsin Dells to Minneapolis Area",
        from: [43.6275, -89.7709],
        to: [44.9778, -93.2650],
        time: "About 3.5-4.5 hr",
        buffer: "30 min",
        do: "Use this as first major break region.",
        skip: "No attractions. This is a driving day.",
        detail: "Breakfast should be quick before departure or early on the road."
      },
      {
        name: "Minneapolis Area to Fargo Area",
        from: [44.9778, -93.2650],
        to: [46.8772, -96.7898],
        time: "About 3.5-4 hr",
        buffer: "30 min",
        do: "Driver switch and fuel if needed.",
        skip: "No long shopping or city detours.",
        detail: "Keep momentum. This is the middle of the mule day."
      },
      {
        name: "Fargo Area to Bismarck",
        from: [46.8772, -96.7898],
        to: [46.8083, -100.7837],
        time: "About 3 hr",
        buffer: "30 min",
        do: "Drive directly to Bismarck, check in, eat, sleep.",
        skip: "No activities in Bismarck.",
        detail: "The goal is arriving rested enough for the Gardiner push tomorrow."
      }
    ]
  },

  sun_sep_6: {
    title: "Sunday, Sept 6 - Bismarck to Gardiner",
    overnight: "Gardiner",
    summary: "Drive to the Yellowstone north entrance area. Mammoth Lower Terraces only if arrival is early and everyone feels good.",
    stops: [
      {
        name: "Bismarck",
        coords: [46.8083, -100.7837],
        note: "Leave around 7:30-8:00 AM Central."
      },
      {
        name: "Billings Area",
        coords: [45.7833, -108.5007],
        note: "Likely meal / fuel region."
      },
      {
        name: "Gardiner",
        coords: [45.0319, -110.7058],
        note: "Check in, eat dinner, fuel the car."
      },
      {
        name: "Mammoth Lower Terraces",
        coords: [44.9769, -110.7013],
        note: "Optional only. Maximum 45-60 minutes."
      }
    ],
    segments: [
      {
        name: "Bismarck to Billings Area",
        from: [46.8083, -100.7837],
        to: [45.7833, -108.5007],
        time: "About 5.5-6.5 hr",
        buffer: "45-60 min",
        do: "Breakfast early, then drive west. Use Billings area for meal/fuel if timing works.",
        skip: "No detours.",
        detail: "This is the long first half of the Gardiner drive."
      },
      {
        name: "Billings Area to Gardiner",
        from: [45.7833, -108.5007],
        to: [45.0319, -110.7058],
        time: "About 2.5-3.5 hr",
        buffer: "30-60 min",
        do: "Check in first, eat dinner, fuel the car.",
        skip: "Do not force Mammoth if arrival is delayed.",
        detail: "Get organized for the early Lamar morning."
      },
      {
        name: "Optional Gardiner to Mammoth Lower Terraces",
        from: [45.0319, -110.7058],
        to: [44.9769, -110.7013],
        time: "Short local drive",
        buffer: "15 min",
        do: "Only if early arrival and everyone feels okay.",
        skip: "Skip if parking is annoying, weather is bad, or people are tired.",
        detail: "Maximum 45-60 minutes. Monday matters more."
      }
    ]
  },
  
  mon_sep_7: {
    title: "Monday, Sept 7 - Lamar Valley + Yellowstone Canyon",
    overnight: "Canyon Village",
    summary: "Early wildlife window first, then Canyon viewpoints. If late, skip Lamar and focus Canyon.",
    stops: [
      {
        name: "Gardiner",
        coords: [45.0319, -110.7058],
        note: "Start before sunrise if doing Lamar Valley."
      },
      {
        name: "Lamar Valley",
        coords: [44.9166, -110.1666],
        note: "Wildlife viewing. Pullouts only. No hiking."
      },
      {
        name: "Canyon Village",
        coords: [44.7358, -110.4893],
        note: "Lunch, reset, lodging area."
      },
      {
        name: "Artist Point",
        coords: [44.7205, -110.4790],
        note: "Main Yellowstone Canyon viewpoint."
      }
    ],
    segments: [
      {
        name: "Gardiner to Lamar Valley",
        from: [45.0319, -110.7058],
        to: [44.9166, -110.1666],
        time: "About 1.5 hr",
        buffer: "30 min",
        do: "Leave around 6:15 AM. Bring breakfast, lunch, water, binoculars, layers.",
        skip: "No hiking. Do not chase animals. Do not get close to wildlife.",
        detail: "Use pullouts and scan with binoculars. Watch where people with scopes are looking."
      },
      {
        name: "Lamar Valley to Canyon Village",
        from: [44.9166, -110.1666],
        to: [44.7358, -110.4893],
        time: "About 2-2.5 hr",
        buffer: "30 min",
        do: "Drive toward Canyon Village through Tower / Dunraven area.",
        skip: "If traffic is bad, skip extra stops and go straight to Canyon.",
        detail: "This is the transition from wildlife morning to Canyon afternoon."
      },
      {
        name: "Canyon Village to Artist Point",
        from: [44.7358, -110.4893],
        to: [44.7205, -110.4790],
        time: "Short local drive",
        buffer: "30 min for parking and walking",
        do: "Artist Point first. Add Upper Falls Viewpoint or one North Rim viewpoint only if convenient.",
        skip: "Skip Brink of Lower Falls if tired. Skip long South Rim hiking.",
        detail: "This is the must-do Canyon section. Do not try every viewpoint."
      }
    ]
  },

  tue_sep_8: {
    title: "Tuesday, Sept 8 - Old Faithful + Grand Prismatic",
    overnight: "Grant Village",
    summary: "Geothermal day. Old Faithful first, short nearby boardwalk, then Grand Prismatic Overlook. West Thumb only if energy is good.",
    stops: [
      {
        name: "Canyon Village",
        coords: [44.7358, -110.4893],
        note: "Leave around 6:30 AM if possible."
      },
      {
        name: "Old Faithful",
        coords: [44.4605, -110.8281],
        note: "Watch Old Faithful and walk nearby Upper Geyser Basin only."
      },
      {
        name: "Grand Prismatic Overlook",
        coords: [44.5146, -110.8320],
        note: "Use Fairy Falls Trailhead. Parking can be limited."
      },
      {
        name: "West Thumb Geyser Basin",
        coords: [44.4155, -110.5733],
        note: "Optional short boardwalk near Grant Village."
      },
      {
        name: "Grant Village",
        coords: [44.3900, -110.5567],
        note: "Overnight stop."
      }
    ],
    segments: [
      {
        name: "Canyon Village to Old Faithful",
        from: [44.7358, -110.4893],
        to: [44.4605, -110.8281],
        time: "About 2-2.5 hr",
        buffer: "30 min",
        do: "Leave early. Go directly to Old Faithful area.",
        skip: "No random pullouts unless necessary.",
        detail: "On arrival, check the next Old Faithful eruption time first."
      },
      {
        name: "Old Faithful area",
        from: [44.4605, -110.8281],
        to: [44.4605, -110.8281],
        time: "About 2-2.5 hr",
        buffer: "45 min",
        do: "Watch Old Faithful, short nearby boardwalk, Visitor Center, Old Faithful Inn lobby if convenient.",
        skip: "Skip Morning Glory Pool. Do not wait for Grand, Castle, or Riverside geysers.",
        detail: "This is not a full Upper Geyser Basin day. Keep it controlled."
      },
      {
        name: "Old Faithful to Grand Prismatic Overlook",
        from: [44.4605, -110.8281],
        to: [44.5146, -110.8320],
        time: "About 45-60 min including parking attempt",
        buffer: "30 min",
        do: "Try Fairy Falls Trailhead parking once. Walk to overlook if parking works.",
        skip: "Skip Midway boardwalk. Skip Fairy Falls full hike. If parking is impossible, move on.",
        detail: "The overlook is the priority because it gives the best view of Grand Prismatic."
      },
      {
        name: "Grand Prismatic to Grant Village",
        from: [44.5146, -110.8320],
        to: [44.3900, -110.5567],
        time: "About 1-1.5 hr",
        buffer: "30 min",
        do: "Drive south toward Grant Village. Add West Thumb only if energy is good.",
        skip: "Skip West Thumb if tired, late, or weather is bad.",
        detail: "Grant Village is the real endpoint. Do not overextend."
      }
    ]
  },

  wed_sep_9: {
    title: "Wednesday, Sept 9 - Yellowstone to Grand Teton",
    overnight: "Moran / Colter Bay",
    summary: "Scenic transfer day. No real hiking. Enjoy Grand Teton without draining energy before the Wall drive.",
    stops: [
      {
        name: "Grant Village",
        coords: [44.3900, -110.5567],
        note: "Start point. Breakfast, pack, and leave around 8:15 AM."
      },
      {
        name: "West Thumb Geyser Basin",
        coords: [44.4155, -110.5733],
        note: "Optional only if not done Tuesday and everyone feels good."
      },
      {
        name: "Colter Bay",
        coords: [43.9044, -110.6418],
        note: "Lunch, bathroom, visitor center if convenient."
      },
      {
        name: "Oxbow Bend",
        coords: [43.8664, -110.5480],
        note: "Scenic stop. Wildlife scan and Mount Moran view."
      },
      {
        name: "Snake River Overlook",
        coords: [43.7540, -110.6247],
        note: "Quick photo stop. No hiking."
      },
      {
        name: "Mormon Row",
        coords: [43.6655, -110.6645],
        note: "Barns and Teton backdrop. Keep it under one hour."
      },
      {
        name: "Moran / Colter Bay",
        coords: [43.8416, -110.5081],
        note: "Check in, dinner, no scheduled sunset."
      }
    ],
    segments: [
      {
        name: "Grant Village to West Thumb",
        from: [44.3900, -110.5567],
        to: [44.4155, -110.5733],
        time: "About 15-20 min",
        buffer: "15 min",
        do: "Stop only if West Thumb was skipped Tuesday and everyone has energy.",
        skip: "Skip if tired, late, or weather is bad.",
        detail: "This is optional. Do not let it hurt the Grand Teton day."
      },
      {
        name: "Grant Village / West Thumb to Colter Bay",
        from: [44.4155, -110.5733],
        to: [43.9044, -110.6418],
        time: "About 2-2.5 hr",
        buffer: "30 min",
        do: "Drive south through Yellowstone South Entrance toward Grand Teton.",
        skip: "No hikes. Stop only for bathroom, fuel, or a truly easy pullout.",
        detail: "This is the transfer from Yellowstone to Grand Teton. Keep it relaxed but controlled."
      },
      {
        name: "Colter Bay to Oxbow Bend",
        from: [43.9044, -110.6418],
        to: [43.8664, -110.5480],
        time: "About 20-30 min",
        buffer: "15 min",
        do: "Take photos and do a wildlife scan.",
        skip: "Do not wait forever for perfect light.",
        detail: "Oxbow Bend is the strongest scenic stop of the day."
      },
      {
        name: "Oxbow Bend to Snake River Overlook",
        from: [43.8664, -110.5480],
        to: [43.7540, -110.6247],
        time: "About 25-35 min",
        buffer: "10 min",
        do: "Quick overlook stop.",
        skip: "No hiking.",
        detail: "This is a short photo stop, not a long activity."
      },
      {
        name: "Snake River Overlook to Mormon Row",
        from: [43.7540, -110.6247],
        to: [43.6655, -110.6645],
        time: "About 25-35 min",
        buffer: "20 min",
        do: "See the barns and Teton backdrop. Easy walking only.",
        skip: "Skip if road or parking delay is annoying.",
        detail: "Keep Mormon Row under one hour so the day stays easy."
      },
      {
        name: "Mormon Row to Moran / Colter Bay",
        from: [43.6655, -110.6645],
        to: [43.8416, -110.5081],
        time: "About 40-60 min",
        buffer: "20 min",
        do: "Check in, dinner, rest.",
        skip: "No scheduled sunset. No Jenny Lake. No Taggart Lake.",
        detail: "The goal is to arrive sane before the long Wall drive tomorrow."
      }
    ]
  },

  thu_sep_10: {
    title: "Thursday, Sept 10 - Grand Teton Sunrise to Wall",
    overnight: "Wall",
    summary: "One sunrise stop only, then drive directly to Wall. No Badlands tonight.",
    stops: [
      {
        name: "Moran / Colter Bay",
        coords: [43.8416, -110.5081],
        note: "Wake up around 5:45-6:15 AM depending on exact sunrise."
      },
      {
        name: "Oxbow Bend",
        coords: [43.8664, -110.5480],
        note: "Sunrise stop only. Photos and quiet wildlife scan."
      },
      {
        name: "Wall",
        coords: [43.9925, -102.2416],
        note: "Check in, dinner, early sleep. Do not enter Badlands tonight."
      }
    ],
    segments: [
      {
        name: "Moran / Colter Bay to Oxbow Bend",
        from: [43.8416, -110.5081],
        to: [43.8664, -110.5480],
        time: "About 15-30 min",
        buffer: "15 min",
        do: "Arrive about 20-30 minutes before sunrise.",
        skip: "Skip Mormon Row sunrise. Skip Jenny Lake. Skip extra viewpoints.",
        detail: "This is the only planned Grand Teton stop today."
      },
      {
        name: "Oxbow Bend to Wall",
        from: [43.8664, -110.5480],
        to: [43.9925, -102.2416],
        time: "About 8.5-10.5 hr total with stops",
        buffer: "1-2 hr",
        do: "Breakfast, fuel, bathroom, then drive directly to Wall.",
        skip: "No bonus stops. No scenic detours. No Badlands this evening.",
        detail: "This is the discipline day. The goal is arriving safely and sleeping early."
      }
    ]
  },

  fri_sep_11: {
    title: "Friday, Sept 11 - Badlands to Wisconsin Dells",
    overnight: "Wisconsin Dells",
    summary: "Short Badlands scenic-drive visit in the morning, then a long drive to Wisconsin Dells.",
    stops: [
      {
        name: "Wall",
        coords: [43.9925, -102.2416],
        note: "Leave around 6:15 AM Mountain."
      },
      {
        name: "Pinnacles Overlook",
        coords: [43.8932, -102.2388],
        note: "First Badlands stop. Quick photos and wildlife scan."
      },
      {
        name: "Yellow Mounds Overlook",
        coords: [43.8338, -102.1752],
        note: "Short geology/photo stop."
      },
      {
        name: "Big Badlands Overlook",
        coords: [43.7484, -101.9415],
        note: "Final scenic stop before the long drive."
      },
      {
        name: "Wisconsin Dells",
        coords: [43.6275, -89.7709],
        note: "Overnight. Expected arrival around 7:30-9:30 PM Central."
      }
    ],
    segments: [
      {
        name: "Wall to Pinnacles Overlook",
        from: [43.9925, -102.2416],
        to: [43.8932, -102.2388],
        time: "About 20-30 min",
        buffer: "10 min",
        do: "Quick sunrise-area scenic stop.",
        skip: "No hiking.",
        detail: "Start the Badlands scenic drive west to east."
      },
      {
        name: "Pinnacles to Yellow Mounds",
        from: [43.8932, -102.2388],
        to: [43.8338, -102.1752],
        time: "About 20-30 min",
        buffer: "10 min",
        do: "Short photo stop.",
        skip: "Do not linger too long.",
        detail: "This is a quick geology/color stop."
      },
      {
        name: "Yellow Mounds to Big Badlands Overlook",
        from: [43.8338, -102.1752],
        to: [43.7484, -101.9415],
        time: "About 30-45 min",
        buffer: "15 min",
        do: "Final scenic stop before leaving the park area.",
        skip: "Skip if already behind schedule.",
        detail: "Badlands is the morning prize. The rest of the day is road."
      },
      {
        name: "Big Badlands Overlook to Wisconsin Dells",
        from: [43.7484, -101.9415],
        to: [43.6275, -89.7709],
        time: "About 9.5-11 hr total with stops",
        buffer: "1-2 hr",
        do: "Drive directly to Wisconsin Dells. Fuel, meal, bathroom only.",
        skip: "No hiking. No extra sightseeing. No long visitor center stop.",
        detail: "You lose one hour going back toward Central Time. Arrive, eat, sleep."
      }
    ]
  },

  sat_sep_12: {
    title: "Saturday, Sept 12 - Wisconsin Dells to Troy",
    overnight: "Home",
    summary: "Final drive home. Leave early, keep stops efficient, and protect the Sunday rest day.",
    stops: [
      {
        name: "Wisconsin Dells",
        coords: [43.6275, -89.7709],
        note: "Leave around 6:30-7:00 AM Central."
      },
      {
        name: "Chicago Area",
        coords: [41.8781, -87.6298],
        note: "Likely traffic / fuel / stretch region."
      },
      {
        name: "Troy",
        coords: [42.6064, -83.1498],
        note: "Home. Expected arrival around 4:30-6:00 PM Eastern."
      }
    ],
    segments: [
      {
        name: "Wisconsin Dells to Chicago Area",
        from: [43.6275, -89.7709],
        to: [41.8781, -87.6298],
        time: "About 3-4 hr",
        buffer: "30-60 min",
        do: "Breakfast stop and efficient fuel/stretch stop.",
        skip: "No sightseeing.",
        detail: "Start early to avoid turning the final day into sludge."
      },
      {
        name: "Chicago Area to Troy",
        from: [41.8781, -87.6298],
        to: [42.6064, -83.1498],
        time: "About 4-5 hr",
        buffer: "30-60 min",
        do: "Drive home. One later fuel/stretch stop if needed.",
        skip: "No bonus stops.",
        detail: "You lose one hour returning to Eastern Time. Sunday is protected rest."
      }
    ]
  }
};