def encode_customer_segment(segment: str):
    is_high_roller = 1 if segment == "High Roller" else 0
    is_luxury_estate = 1 if segment == "Luxury Estate" else 0
    is_wine_enthusiast = 1 if segment == "Wine Enthusiast" else 0
    return [is_high_roller, is_luxury_estate, is_wine_enthusiast]

def encode_division(division: str):
    is_east_south_central = 1 if division == "East South Central" else 0
    is_middle_atlantic = 1 if division == "Middle Atlantic" else 0
    is_mountain = 1 if division == "Mountain" else 0
    is_new_england = 1 if division == "New England" else 0
    is_pacific = 1 if division == "Pacific" else 0
    is_south_atlantic = 1 if division == "South Atlantic" else 0
    is_west_north_central = 1 if division == "West North Central" else 0
    is_west_south_central = 1 if division == "West South Central" else 0

    return [is_east_south_central, is_middle_atlantic, is_mountain,
             is_new_england, is_pacific, is_south_atlantic,
               is_west_north_central, is_west_south_central]
