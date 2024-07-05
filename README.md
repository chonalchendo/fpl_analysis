# Project Name

Football Player Value Prediction App

## Description

This project is a web application that predicts the value of a football player based on performance attributes for the 2023/24 season.

The model used is a RandomForestRegressor which is trained on data from the 2017/18 to 2022/23 seasons.

The user can filter the results dataframe based on the player country, position, and league via the web application's user interface built using Streamlit.

## Table of Contents

[Installation](#installation)
[Usage](#usage)

## Installation

1. Clone the repository

```bash
git clone https://github.com/chonalchendo/fpl_analysis.git
```

2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

For further instructions on how to install Poetry, visit the [Poetry documentation - installation](https://python-poetry.org/docs/#installing-with-the-official-installer).

3. Install Docker

The Docker version used for this project is v4.32.0

[Linux](https://docs.docker.com/desktop/install/linux-install/)
[Mac](https://docs.docker.com/desktop/install/mac-install/)
[Windows](https://docs.docker.com/desktop/install/windows-install/)

## Usage

1. Run the production Docker container

```bash

docker compose -f deploy/app-docker-compose.yml --project-directory . up --build
```

2. Run the development Docker container

```bash
docker compose -f deploy/app-docker-compose.yml -f deploy/app-docker-compose.local.yml --project-directory . up --build
```

Running this will allow you to reload the web app in real-time as you make changes to the code.

3. Access the web application

The web application can be accessed at [http://localhost:8501](http://localhost:8501)
FastAPI can be queried directly at [http://localhost:8000](http://localhost:8000)

## Screenshots

![Landing Page](/Users/conal/Projects/fpl_app/images/fpl_greeting_page.png)
![Predictions Page](/Users/conal/Projects/fpl_app/images/fpl_query_result.png)

## Technologies Used

- Python
- Streamlit
- FastAPI
- Docker
- Poetry
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- httpx
- Pydantic
- Uvicorn
- GCP

## Licensing Information

MIT

## Roadmap

- Currently we only have half of the 2023/24 season data which is likely causing predictions to be poor as the model is trained on full season data. We will need to update the model with the rest of the season data.
- Built out the webscraper to get more data.
- Create a model that draws on all data sources to make predictions rather than just attacking stats.
- Built a probabilitistic model that predicts player performance.
- Add new player performance related visualisations to frontend.
- Look at building a JavaScript frontend to map the app more professional and interactive.

## Links

Connect with me on [LinkedIn](https://www.linkedin.com/in/conal-henderson-4128631b6/)!
