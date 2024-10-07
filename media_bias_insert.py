import json


class SentinelEmpty:
    """
    Sentinel indicating that the location it occupies is empty.
    """

    def __repr__(self):
        return "<Empty>"


RAW_CLAIM_DATA = 'data/media_bias.json'
EMPTY_SENTINEL = SentinelEmpty()

# Logging
LOGGING = False


def get_name_of_url(url: str) -> str:
    # Note we observed no urls starting with "http"
    assert url[0:4] != "http"

    # Strip leading 'www.'
    if url[0:4] == "www.":
        url = url[4:]

    # Only want domain info
    url = url.split('/')[0]
    assert len(url) != 0

    # Discard domain extension (i.e. '.com')
    if "." in url:
        url = ''.join(url.split(".")[0:-1])

    return url


def get_data_by_url(file):
    data: list = json.load(file)
    data_by_url: dict = {}
    data_to_remove: set = set()
    for entry in data:
        if entry['url'] == 'no url available' or entry['bias'] == 'No media bias rating':
            continue

        url = get_name_of_url(entry['url'])
        bias = entry['bias'].replace("-", "_").upper()

        existing_entry_bias = data_by_url.get(url, EMPTY_SENTINEL)
        # Check if entry has an existing entry
        if existing_entry_bias != EMPTY_SENTINEL:
            # Check if existing entry has conflicting value
            if existing_entry_bias != bias:
                # If conflicting entries, then we can't decide so remove all entries
                data_to_remove.add(url)
        else:
            data_by_url[url] = bias

    for url in data_to_remove:
        data_by_url.pop(url)

    if LOGGING:
        print(f"Removed {len(data_to_remove)}")

    return data_by_url



def main():
    pass


if __name__ == '__main__':
    main()
