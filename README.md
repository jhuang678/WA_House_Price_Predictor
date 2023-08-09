# DVA Group Project: No Place Like Home 
## Description
This is the python-Flask package for the **No Place Like Home** application, an application that helps users find 
places to live, initially for the state of Georgia by zipcode. To use the application, you can install and launch it by
running the flask app locally from the repo or using docker to create a container of the flask application.

The application consist of 3 sections:
1. Data/Backend
   1. Data Sources
      1. DATA.GOV (https://data.gov/)
      2. Federal Bureau of Investigation Crime Data Explorer (https://cde.ucr.cjis.gov/)
      3. United State Census Bureau Data (https://data.census.gov/)
      4. Bestplaces (https://www.bestplaces.net/)
   2. Data Scraping: 
      1. The data scraping application is written in Python - `web_crawler_ga_zip5.py`
      2. If you want to run it, we recommend running the script through a local IDE since running on a terminal may not pass the SSL verification.
   3. Data Cleaning...
   4. Result of the backend is stored as a flat csv file.
2. Front-end
   1. The application uses Flask to host the navigation and logic of the server.
   2. Flask-WTForm is used for creating forms.
   3. Bootstrap is used as the styling library.
   4. Folium is used for the interactive map interface.
3. Algorithm
   1. The recommendation algorithm is written in Python
   2. It uses distance-based measures to calculate similarity and ranks the available locations based 
on user targets and preferences.

## Installation
### Option 1: Run locally (for team member only)
#### Prereq
1. Pull project from github.
2. Create a virtual environment. (https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
```commandline
py -m venv env
.\env\Scripts\activate
```
3. Install dependency library based on requirements.txt
```commandline
py -m pip install -r requirements.txt
```
4. Download prepared data from google drive
https://drive.google.com/uc?export=download&id=1RRY6m_BvwAVxEPjfYYlefALS7Ug9fJda
7. Place the file under the data folder as `./data/demo_geo.csv`.

#### Run Flask App
1. Option 1: Run the `app.py` script. It should launch a flask app in ```DEBUG``` mode.
2. Option 2: Run the flask app in command line.
```commandline
flask --app app run
```
(debug mode)
```commandline
flask --app hello run --debug
```
#### Close Application
Terminate Flask app by ctrl+C
### Option 2: Run with Docker
Youtube Demo Link: https://www.youtube.com/watch?v=a0F7bs_xGNY
1. Install docker and/or docker desktop
2. Login to docker (replacing `YOUR-USER-NAME` with your username for docker)
```commandline
docker login -u YOUR-USER-NAME
```
3. Pull docker image to your local machine. Docker hub image is located here: https://hub.docker.com/r/garyyen82/team50 
```commandline
docker pull garyyen82/team50:v2
```
4. Run the docker image
```commandline
docker run -p 5000:5000 garyyen82/team50:v2
```
5. Access the Flask application by navigating to http://127.0.0.1:5000
#### Close Application
1. To shutdown container, run this on a new terminal.
```commandline
docker stop <container_name_or_id>
```
If you don't know the container id, check by running this.
```commandline
docker ps
```
2. To stop and remove container, run this.
```commandline
docker rm -f <container_name_or_id>
```
## Execution
Here is an example of running a demo of the application.
### Explore
1. The home page is the explore functionality. If you are on another page, you can get to the Explore page 
by clicking on the `Explore` navigation button at the top.
2. Select `ZIP5` for Level of Aggregation. Currently this is the only available option and the default.
3. Select `Violent Crime Index` for Feature.
4. Select a display color of choice. The default is a gold-ish color resembling the color of Georgia Tech.
5. Adjust the display opacity as needed. The default is 0.5.
6. Click the `Submit` button to refresh the map accordingly.
7. Clicking on the polygons on the map, you can see details about the zipcode area.
### Filter
1. Navigate to the Filter page by clicking on the `Filter` navigation button at the top.
2. Enter `300` to the Zipcode Filter box.
3. Click the `Apply Filters` button to filter location down to zipcode starting with `300`.
4. Clicking on the polygons on the map, you can see additional details about hte zipcode area.
### Recommender
1. Navigate to the Recommender page by clicking on the `Recommender` navigation button at the top.
2. Click the `Filter` box on the left to expand the Filter form. Use this form just like the filter page to
filter down the locations set before the recommendation algorithm takes place.
3. Click `Apply Filters` once the desired Filter input is entered. For example, use `1000000` for `Max` `Median Housing 
Price`, `10` for `Min` `UTCI`, and 20 for `Max` `UTCI`.
4. Click the `Recommend` box on the left to expand the Recommender form. Use this form to fill out the user preferred
values for each feature and their feature importance. For example, use `450000` for `Median Housing Price` and select
`strong` importance, `4` for `Violent Crime Index` and select `some` importance, `8` for `Property Crime Index` and 
select `some` importance, and `0.3` for `Race Share - Asian` and select `strong` for importance.
5. Click `Submit` to start the algorithm.
6. The map will refresh to filter the location polygons down to the top 10 recommended by the algorithm. Each of them
are labeled with the ranking number. You can click on each one of them to get more details about the location.
7. Click `Result` box on the left to expand the Result form. This contains a tabulated result of the recommended 
locations with their respective ranking and similarity score.




