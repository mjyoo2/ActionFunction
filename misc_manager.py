
def __INIT_PATH():
    default_path = input("please input default path ")
    try:
        with open("/misc/TENSORBOARDPATH.txt", "w") as f:
            f.write(default_path)
    except:
        print("ERROR IN SETTING DEFAULT PATH")
        exit(-1)
    return


