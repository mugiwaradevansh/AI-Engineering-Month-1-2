from typing import Any

import requests

BASE_URL = "https://jsonplaceholder.typicode.com"

TIMEOUT = 10

def get_all_users() ->None:
    #Fetch all users and print selected fields.
    response = requests.get(
        f"{BASE_URL}/users",
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    users:list[dict[str,Any]] = response.json()

    print("\n === ALL USERS ===")

    for user in users:
        print(
            f"Name:{user['name']} | "
            f"Email: {user['email']} |"
            f"City:{user['address']['city']}"
        )

def get_single_post(post_id:int) ->None:
    #Fetch one post and print its details.
    response = requests.get(
        f"{BASE_URL}/post/{post_id}",
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    post:dict[str,Any] = response.json()

    print("\n=== SINGLE POST ===")
    print(f"Post ID:{post['id']}")
    print(f"Title:{post['title']}")
    print(f"Body:{post['body']}")

def get_comments_by_post(post_id:int) ->None:
    #Fetch comments for a post and print selected fields
    response =requests.get(
        f"{BASE_URL}/comments",
        params = {"postID":post_id},
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    comments:list[dict[str,Any]] = response.json()

    print(f"\n=== COMMENTS FOR POST {post_id} ===")

    for comment in comments:
        print(f"Name:{comment['name']}")
        print(f"Email:{comment['email']}")
        print(f"Comment:{comment['body']}")
        print("-"* 40)

def main() ->None:
    try:
        get_all_users()
        get_single_post(1)
        get_comments_by_post(1)

    except requests.RequestException as error:
        print(f"Could not parse the response:{error}")

if __name__ == "__main__":
    main()