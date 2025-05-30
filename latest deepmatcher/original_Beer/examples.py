Beer_Fewshot_exampels=f"""
### EXAMPLES OF GOOD STANDARDIZATION:

## Example 1:
Input 1:
Beer Name: Red Amber Ale
Brewery: Example Brewing
Style: American Amber / Red Ale
ABV: 5.5%

Standardized Output:
{{
  "name": "Red Amber Ale",
  "brewery": "Example Brewing",
  "primary_style": "American Amber",
  "secondary_style": "Amber",
  "abv": 5.5,
  "is_amber": "true",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "false",
  "special_ingredients": "none"
}}

## Example 2: Style breakdown for IPAs
Input 1:
Beer Name: Hazy Double IPA
Brewery: Craft Brewers
Style: New England Double India Pale Ale
ABV: 8.2%

Standardized Output :
{{
  "name": "Hazy Double IPA",
  "brewery": "Craft Brewers",
  "primary_style": "IPA",
  "secondary_style": "Double Hazy",
  "abv": 8.1,
  "is_amber": "false",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "true",
  "special_ingredients": "none"
}}

## Example 3: Style breakdown for Stouts
Input 1:
Beer Name: Chocolate Coffee Stout
Brewery: Dark Brewing Co.
Style: American Imperial Stout
ABV: 10.5%

Standardized Output (for both):
{{
  "name": "Chocolate Coffee Stout",
  "brewery": "Dark Brewing",
  "primary_style": "Stout",
  "secondary_style": "Imperial",
  "abv": 10.5,
  "is_amber": "false",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "true",
  "special_ingredients": "chocolate, coffee"
}}

## Example 4: Matching beers with different style descriptions
Input 1:
Beer Name: Sanibel Red Island Ale
Brewery: Point Ybel Brewing Company
Style: American Amber / Red Ale
ABV: 5.60%


Standardized Output (for both):
{{
  "name": "Sanibel Red Island Ale",
  "brewery": "Point Ybel Brewing",
  "primary_style": "Red Ale",
  "secondary_style": "Amber",
  "abv": 5.6,
  "is_amber": "true",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "false",
  "special_ingredients": "none"
}}

## Example 5: Style breakdown for barrel-aged beers
Input:
Beer Name: Buffalo Trace Bourbon Barrel Aged Amber Ale
Brewery: Wolf Hills Brewing Company
Style: American Amber / Red Ale
ABV: 7.5%

Standardized Output:
{{
  "name": "Buffalo Trace Bourbon Barrel Aged Amber Ale",
  "brewery": "Wolf Hills Brewing",
  "primary_style": "Amber Ale",
  "secondary_style": "Barrel-Aged",
  "abv": 7.5,
  "is_amber": "true",
  "is_ale": "true",
  "is_lager": "false",
  "is_imperial": "false",
  "special_ingredients": "bourbon"
}}

"""
