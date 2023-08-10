######################################################################
import requests as req
from bs4 import BeautifulSoup
import time
import csv
import re
################################################R######################

if __name__ == "__main__":
    URL = "https://www.bestplaces.net/find/state.aspx?state=wa"
    response = req.get(URL)

    # Create a BeautifulSoup object
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the div with class 'row mt-3'
    row_div = soup.find('div', {'class': 'row mt-3'})

    # Find all the u tags within zip_div
    u_tags = row_div.find_all('u')

    # Extract the text from the u tags and store it in a list
    cities = [u.text for u in u_tags]

    # Create a list to store the data
    data = []

    for city in cities[292:]:
        print("Number of city: ", len(data)+1)
        print("Name of City: ", city)

        # Extract the zip code and city name
        city_lower = city[:].replace(' ', '_').lower()  # Convert to lowercase and replace spaces with underscores
    
        # Overview
        link = f"https://www.bestplaces.net/city/washington/{city_lower}"
        response = req.get(link)
        soup = BeautifulSoup(response.content, "html.parser")

        p_elements = soup.find_all('p', {'class': 'text-center'})

        for p in p_elements:
            if 'population' in p.text.lower():
                population = p.find_next_sibling('p').text.replace(',', '')
                #print(f"Population: {population}")

                pattern = r"-?\d+\.\d+%"
                migration_2020_text = re.search(pattern, p.find_next_sibling('p').find_next_sibling('p').text).group()
                migration_2020 = round(float(migration_2020_text.strip('%'))/100,4)
                #print(f"Migration rate since 2022: {migration_2020}")

            if 'median income' in p.text.lower():
                median_income = p.find_next_sibling('p').text.strip('$').replace(',', '')
                #print(f"Median Income: {median_income}")

            if 'median age' in p.text.lower():
                median_age = p.find_next_sibling('p').text
                #print(f"Median Age: {median_age}")

            if 'unemployment rate' in p.text.lower():
                unemployment_rate_text = p.find_next_sibling('p').text
                unemployment_rate = round(float(unemployment_rate_text.strip('%'))/100, 4)
                #print(f"Unemployment Rate: {unemployment_rate}")

            if 'median home price' in p.text.lower():
                median_home_price = p.find_next_sibling('p').text.strip('$').replace(',', '')
                #print(f"Median home price: {median_home_price}")

            if 'comfort index' in p.text.lower():
                comfort_index = p.find_next_sibling('p').text.split("/")
                summer_comfort_index = float(comfort_index[0].strip())
                winter_comfort_index = float(comfort_index[1].strip())

                #print(f"Summer Comfort Index: {summer_comfort_index}")
                #print(f"Winter Comfort Index: {winter_comfort_index}")


        div_elements = soup.find_all('div', {'class': 'col-md-12'})
        for div in div_elements:

            if 'cost of living is' in div.text:
                try:
                    match = re.search(r'(\d+\.\d+)% (lower|higher)', div.text)
                    value = float(match.group(1))
                    if match.group(2) == "lower":
                        cost_of_living_index = round((100 - value) / 100,4)
                    else:
                        cost_of_living_index = round((100 + value) / 100,4)
                    #print(f"Cost of Living Index: {cost_of_living_index}")
                except AttributeError:
                    cost_of_living_index = ''

            if 'Commute time' in div.text:
                match = re.search(r'Commute time is (\d+\.\d+) minutes', div.text)
                mean_commute_time = float(match.group(1))
                #print(f"Average Commute Time: {mean_commute_time}")

            if 'public schools spend' in div.text:
                try:
                    match = re.search(r'public schools spend \$(\d+\,\d+) per student', div.text)
                    mean_school_expense = float(match.group(1).replace(',', ''))
                    #print(f"Average Student expenditure: {mean_school_expense}")
                except AttributeError:
                    mean_school_expense = ''


            if 'students per teacher' in div.text:
                try:
                    match = re.search(r'There are about (\d+(?:\.\d+)?) students per teacher', div.text)
                    student_per_teacher = float(match.group(1))
                    # print(f"Average Number of Student per Teacher: {student_per_teacher}")
                except AttributeError:
                    student_per_teacher = ''

        # Crime
        link = f"https://www.bestplaces.net/crime/city/washington/{city_lower}"
        response = req.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        vc_heading = soup.select_one('h5:-soup-contains("violent crime")')
        vci = '' if vc_heading is None else re.search(r'(\d+\.\d+)', vc_heading.get_text(strip=True)).group(1)
        pc_heading = soup.select_one('h5:-soup-contains("property crime")')
        pci = '' if pc_heading is None else re.search(r'(\d+\.\d+)', pc_heading.get_text(strip=True)).group(1)
        #print(f"Violent Crime Index: {vci}")
        #print(f"Property Crime Index: {pci}")

        # Economic
        link = f"https://www.bestplaces.net/economy/city/washington/{city_lower}"
        response = req.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        div_elements = soup.find_all('div', {'class': 'col-md-12'})
        sales_tax_rate = ''
        income_tax_rate = ''
        for div in div_elements:
            if 'The Sales Tax Rate' in div.text:
                try:
                    match = re.search(r'The Sales Tax Rate for ' + city + r' is (\d+\.\d+)%', div.text)
                    sales_tax_rate = round(float(match.group(1))/100, 4)
                    #print(f"Sales Tax Rate: {sales_tax_rate}")
                except AttributeError:
                    sales_tax_rate = ''

            if 'The Income Tax Rate' in div.text:
                try:
                    match = re.search(r'The Income Tax Rate for ' + city + r' is (\d+\.\d+)%', div.text)
                    income_tax_rate = round(float(match.group(1)) / 100, 4)
                    #print(f"Income Tax Rate: {income_tax_rate}")
                except AttributeError:
                    income_tax_rate = ''

        # Health
        link = f"https://www.bestplaces.net/health/city/washington/{city_lower}"
        response = req.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        div_elements = soup.find_all('div', {'class': 'display-4'})

        try:
            text = div_elements[0].text.split("/")[0]
            health_cost_index = round(float(text)/100,4)
            #print(f"Health Cost Index: {health_cost_index}")
        except IndexError:
            health_cost_index = ''

        try:
            text = div_elements[1].text.split("/")[0]
            water_quality_index = round(float(text) / 100, 4)
            #print(f"Water Quality Index: {water_quality_index}")
        except IndexError:
            water_quality_index = ''

        try:
            text = div_elements[3].text.split("/")[0]
            air_quality_index = round(float(text) / 100, 4)
            #print(f"Air Quality Index: {air_quality_index}")
        except IndexError:
            air_quality_index = ''

        # People
        link = f"https://www.bestplaces.net/people/city/washington/{city_lower}"
        response = req.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        try:
            pd_text = soup.find('b', string=re.compile('people per square mile')).text
            population_density = re.search('(\d+(,\d+)*)', pd_text).group(1).replace(',', '')
            population_density = float(population_density)
        except AttributeError:
            population_density = ''

        table = soup.find('table', {'id': 'mainContent_dgPeople'})
        rows = table.find_all('tr')

        # Create a dictionary to store the race percents
        tb_dict = {}

        # Loop through the rows and extract the race percents
        for row in rows:
            cells = row.find_all('td')
            if len(cells) == 3:
                race = cells[0].get_text().strip()
                percent = cells[1].get_text().strip()
                tb_dict[race] = percent

        # Assign the race percents
        white_percent = ''
        white_value = tb_dict.get('White', None)
        if white_value is not None:
            white_percent = round(float(white_value.strip('%')) / 100, 4)

        black_percent = ''
        black_value = tb_dict.get('Black', None)
        if black_value is not None:
            black_percent = round(float(black_value.strip('%')) / 100, 4)

        asian_percent = ''
        asian_value = tb_dict.get('Asian', None)
        if asian_value is not None:
            asian_percent = round(float(asian_value.strip('%')) / 100, 4)

        native_american_percent = ''
        native_american_value = tb_dict.get('Native American', None)
        if native_american_value is not None:
            native_american_percent = round(float(native_american_value.strip('%')) / 100, 4)

        pacific_islander_percent = ''
        pacific_islander_value = tb_dict.get('Hawaiian, Pacific Islander', None)
        if pacific_islander_value is not None:
            pacific_islander_percent = round(float(pacific_islander_value.strip('%')) / 100, 4)

        other_percent = ''
        other_value = tb_dict.get('Other Race', None)
        if other_value is not None:
            other_percent = round(float(other_value.strip('%')) / 100, 4)

        multiple_percent = ''
        multiple_value = tb_dict.get('Two or More Races', None)
        if multiple_value is not None:
            multiple_percent = round(float(multiple_value.strip('%')) / 100, 4)

        hispanic_percent = ''
        hispanic_value = tb_dict.get('Hispanic', None)
        if hispanic_value is not None:
            hispanic_percent = round(float(hispanic_value.strip('%')) / 100, 4)

        # Data Collection
        data.append({'city': city,
                     'population': population,
                     'migration_2020': migration_2020,
                     'unemployment_rate': unemployment_rate,
                     'median_income': median_income,
                     'median_age': median_age,
                     'summer_comfort_index': summer_comfort_index,
                     'winter_comfort_index': winter_comfort_index,
                     'cost_of_living_index': cost_of_living_index,
                     'mean_commute_time': mean_commute_time,
                     'mean_school_expense': mean_school_expense,
                     'student_per_teacher': student_per_teacher,
                     'violent_crime_index': vci,
                     'property_crime_index': pci,
                     'sales_tax_rate': sales_tax_rate,
                     'income_tax_rate': income_tax_rate,
                     'health_cost_index': health_cost_index,
                     'water_quality_index': water_quality_index,
                     'air_quality_index': air_quality_index,
                     'white_percent': white_percent,
                     'black_percent': black_percent,
                     'asian_percent': asian_percent,
                     'native_american_percent': native_american_percent,
                     'pacific_islander_percent': pacific_islander_percent,
                     'other_percent': other_percent,
                     'multiple_percent': multiple_percent,
                     'hispanic_percent': hispanic_percent,
                     'population_density': population_density,
                     'median_house_price': median_home_price
                     })

    # Write the data to a CSV file
    fieldnames = list(data[0].keys())

    with open('city_in_washington.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print("Done!")
