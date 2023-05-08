
import cv2



def main():
    # 导入MP4文件
    video = cv2.VideoCapture('ecog/Walk.mp4')
    print(type(video))

    # 获取视频的帧率
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    # 获取视频的总帧数
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame_count)

    # 将视频转换为矩阵
    ret, frame = video.read()
    print(type(frame))
    print(frame.shape)

    # 截图第五帧
    cut_frame = frame[,5,:]
    # 展现截图
    cv2.imshow('frame', cut_frame)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()





if __name__ == '__main__':
    main()