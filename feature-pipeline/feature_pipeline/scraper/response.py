import requests
from feature_pipeline.core.settings import SETTINGS
from feature_pipeline.utilities.utils import get_logger

logger = get_logger(__name__)


def get_url_data(url: str) -> requests.models.Response:
    """Scrape data from a url using the scraperapi.com service.

    Args:
        url (str): url to scrape

    Returns:
        requests.models.Response: response object from scraperapi.com
    """
    payload = {
        "api_key": SETTINGS["WEB_SCRAPING_API_KEY"],
        "url": url,
        # "render_js": "false",
    }
    try:
        r = requests.get(SETTINGS["WEB_SCRAPING_API"], params=payload)
        return r
    except requests.exceptions.RequestException as e:
        logger.error(e)


def alt_get_url_data(url: str) -> requests.models.Response:
    headers = {"apikey": SETTINGS["ZENSCRAPER_API_KEY"]}

    params = (("url", url),)

    try:
        r = requests.get(SETTINGS["ZENSCRAPER_API"], headers=headers, params=params)
        logger.info(f"Request status: {r.status_code}")
        return r
    except requests.exceptions.RequestException as e:
        logger.error(e)
