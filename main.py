import os
import sys
import shutil
import subprocess
import platform
import RF_combined, MLP_combined, Global_merge, Boxplot, Algo_count_summary, merge_csv

def main():
    RF_combined.RF()
    MLP_combined.MLP()
    Global_merge.Merge()
    Boxplot.boxplot()
    Algo_count_summary.summary()
    merge_csv.tables()


def notify_beep():
    system = platform.system()

    if system == "Windows":
        try:
            import winsound
            # 方案1：播放系统默认提示音
            winsound.MessageBeep()
            return True
        except Exception:
            pass

        try:
            import winsound
            # 方案2：主动发一个蜂鸣
            winsound.Beep(1000, 300)
            return True
        except Exception:
            return False

    elif system == "Darwin":  # macOS
        # 系统自带声音文件
        sound_file = "/System/Library/Sounds/Ping.aiff"
        if os.path.exists(sound_file):
            player = shutil.which("afplay")
            if player:
                try:
                    subprocess.run(
                        [player, sound_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    return True
                except Exception:
                    pass

        # 退化为终端 bell
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
            return True
        except Exception:
            return False

    elif system == "Linux":
        # 方案1：终端 bell
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass

        # 方案2：如果有 canberra-gtk-play，就播放事件音
        player = shutil.which("canberra-gtk-play")
        if player:
            try:
                subprocess.run(
                    [player, "-i", "bell"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                return True
            except Exception:
                pass

        # 方案3：如果有 paplay，也可以尝试系统声音
        player = shutil.which("paplay")
        candidates = [
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
            "/usr/share/sounds/freedesktop/stereo/complete.oga",
        ]
        if player:
            for sound_file in candidates:
                if os.path.exists(sound_file):
                    try:
                        subprocess.run(
                            [player, sound_file],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                        return True
                    except Exception:
                        pass

        # bell 至少已经尝试过
        return True

    return False



if __name__ == "__main__":
    main()
    notify_beep()