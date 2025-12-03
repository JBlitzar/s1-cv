import subprocess
import time
import uuid

def save_5s(url):
    id = str(uuid.uuid4())
    print("Processing URL:", url, "with ID:", id)
    # download video
    subprocess.run([
        'ffmpeg', '-i', url, '-t', '5', '-c:v', 'libx264', '-c:a', 'aac', f'data/{id}.mp4'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # get last frame
    subprocess.run([
        'ffmpeg', '-sseof', '-3', '-i', f'data/{id}.mp4', '-update', '1', '-q:v', '1', f'data/{id}_last_frame.png'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # get last frame, again
    # that way it's easy to draw over in gimp
    subprocess.run([
        'ffmpeg', '-sseof', '-3', '-i', f'data/{id}.mp4', '-update', '1', '-q:v', '1', f'data/masks/{id}_mask.png'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



# commented returns 404 ??
urls = [
    #"https://wzmedia.dot.ca.gov/D4/S280_on_Monterey_Bl.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/N280_at_Indiana_St.stream/playlist.m3u8",
    #"https://wzmedia.dot.ca.gov/D4/N101_NOF_Willow_Rd.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/N101_at_Marsh_Rd.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/S101_NOF_84_Woodside_Rd.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/S101_at_Whipple_Av.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/N101_JSO_E_Hilldale_Bl.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/W92_at_El_Camino_Real.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/S101_JSO_San_Bruno_Av.stream/playlist.m3u8",
    "https://wzmedia.dot.ca.gov/D4/S101_at_280_Split.stream/playlist.m3u8"
]
if __name__ == "__main__":
    save_5s(urls[0])
    exit()
    # import concurrent.futures

    # for _ in range(3):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(save_5s, url) for url in urls]
    #         for future in concurrent.futures.as_completed(futures):
    #             future.result()
    #     time.sleep(5)
