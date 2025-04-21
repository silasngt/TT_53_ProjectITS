# Object Detecion
import math
import numpy as np
import cv2
import tempfile
from PIL import Image
import datetime
import createBB_helmet
from sort import Sort
from ultralytics import YOLO
from testLane import draw_text, imageViolateHelmet


# hàm để lưu vi phạm mũ bảo hiểm vào database
def create_helmet_violation(mysql_connection, cls):
    """
    Lưu vi phạm mũ bảo hiểm vào database

    Parameters:
        mysql_connection: Kết nối MySQL
        cls: Loại phương tiện vi phạm (0: không mũ bảo hiểm, 1: có mũ bảo hiểm)
    """
    if mysql_connection:
        try:
            cur = mysql_connection.cursor()
            ngay_hien_tai = datetime.date.today()
            # Sử dụng id_name=2 cho xe máy trong bảng nametransportation
            cur.execute(
                "INSERT INTO transportationviolation(id_name, date_violate) VALUES (%s, %s)",
                (2, ngay_hien_tai)  # id_name=2 là "Xe May" trong bảng nametransportation
            )
            mysql_connection.commit()
            cur.close()
            print(f"Đã lưu vi phạm mũ bảo hiểm vào database cho xe máy")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu vi phạm mũ bảo hiểm vào database: {e}")
            return False
    return False


def risize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


def video_detect_helmet(path_x, mysql_connection=None):
    """
    Phát hiện vi phạm mũ bảo hiểm từ video

    Parameters:
        path_x: Đường dẫn video
        mysql_connection: Kết nối MySQL để lưu vi phạm
    """
    print(f"Đang mở video: {path_x}")
    cap = cv2.VideoCapture(path_x)  # For video

    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print(f"Không thể mở video: {path_x}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Khong the mo video", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        yield error_frame
        return

    examBB = createBB_helmet.infoObject()

    try:
        model = YOLO("model_helmet/helmet.pt")  # large model works better with the GPU
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Loi khi tai model", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        yield error_frame
        return

    # tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    dataBienBan_XEMAYVIPHAMBAOHIEM = "F:/python_project/BienBanNopPhatXeMayViPhamMuBaoHiem/ "
    name_class = ["without helmet", "helmet"]
    array_helmet_filter = []
    count = 0
    frame_count = 0

    while True:
        # Kiểm tra nếu đọc frame thành công
        success, frame = cap.read()
        if not success:
            print(f"Đã xử lý hết video hoặc lỗi đọc frame: {frame_count} frames đã được xử lý")
            break

        frame_count += 1
        print(f"Đang xử lý frame #{frame_count}")

        # Tạo bản sao của frame để tránh ghi đè
        display_frame = frame.copy()

        try:
            results = model(frame, stream=True)
            detections = np.empty((0, 6))

            for r in results:
                boxes = r.boxes
                name = r.names
                for box in boxes:
                    # BBOX
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    bbox = (x1, y1, w, h)
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = model.names[cls]

                    # Thu thập thông tin cho mọi đối tượng
                    currentArray = np.array([x1, y1, x2, y2, conf, cls])
                    detections = np.vstack((detections, currentArray))

            classes_array = detections[:, -1:]
            resultsTracker = tracker.update(detections)

            try:
                if len(resultsTracker) > 0:
                    resultsTracker = np.hstack((resultsTracker, classes_array))
            except ValueError:
                if len(resultsTracker) > 0:  # Kiểm tra nếu resultsTracker không rỗng
                    classes_array = classes_array[: resultsTracker.shape[0], :]
                    resultsTracker = np.hstack((resultsTracker, classes_array))

            # Vẽ ROI trước để nằm dưới các annotation khác
            start_point = (0, int((2 * display_frame.shape[0]) / 10))
            end_point = (
                int(display_frame.shape[1]),
                int((8 * display_frame.shape[0]) / 10),
            )
            color = (255, 0, 0)
            cv2.rectangle(display_frame, start_point, end_point, color, 2)

            for result in resultsTracker:
                x, y, w, h, id, cls = result
                x, y, w, h, id, cls = int(x), int(y), int(w), int(h), int(id), int(cls)

                text = str(id) + ": " + name_class[cls]

                center_x = (x + w) // 2
                center_y = (y + h) // 2

                filterData = 0 <= center_x <= (int(display_frame.shape[1])) and int(
                    3 * display_frame.shape[0] / 10
                ) <= center_y <= int(4 * display_frame.shape[0] / 10)

                # Chỉ hiển thị khi đối tượng nằm trong vùng ROI
                if 0 < center_x < int(display_frame.shape[1]) and int(
                        (2 * display_frame.shape[0]) / 10
                ) < center_y < int((8 * display_frame.shape[0]) / 10):
                    # Vẽ hình chữ nhật và hiển thị nhãn khi nằm trong ROI
                    cv2.rectangle(display_frame, (x, y), (w, h), (36, 255, 12), 2)
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Hiển thị nhãn dựa trên class
                    if cls == 1:  # helmet
                        draw_text(
                            display_frame,
                            "Helmet",
                            font_scale=0.5,
                            pos=(int(x), int(y)),
                            text_color=(26, 93, 26),
                            text_color_bg=(208, 192, 79),
                        )
                    else:  # without helmet
                        draw_text(
                            display_frame,
                            text + " warning",
                            font_scale=0.5,
                            pos=(int(x), int(y)),
                            text_color_bg=(0, 0, 0),
                        )

                    # Xử lý vi phạm nếu không đội mũ bảo hiểm
                    if filterData and id not in array_helmet_filter and cls == 0:
                        count += 1
                        array_helmet_filter.append(id)

                        # Lưu ảnh vi phạm
                        imageViolateHelmet(
                            frame,
                            int((0 * frame.shape[0]) / 10),
                            int((8 * frame.shape[0]) / 10),
                            0 * int(frame.shape[1] / 10),
                            8 * int(frame.shape[1] / 10),
                            id,
                        )

                        # Tạo biên bản vi phạm
                        stt_BB_CTB = dataBienBan_XEMAYVIPHAMBAOHIEM + str(id) + ".pdf"
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # Tạo tệp tạm thời và lưu ảnh PIL vào đó
                        temp_image = tempfile.NamedTemporaryFile(
                            suffix=".jpg", delete=False
                        )
                        frame_pil.save(temp_image.name)
                        createBB_helmet.bienBanNopPhat(
                            examBB,
                            temp_image.name,
                            "F:/python_project/data_xe_vp_bh/ " + str(id) + ".jpg",
                            stt_BB_CTB,
                        )
                        temp_image.close()

                        # Lưu vi phạm vào database
                        if mysql_connection:
                            create_helmet_violation(mysql_connection, 0)  # 0 là cho vi phạm không mũ bảo hiểm

                        print(f"Vi phạm #{count} - ID xe: {id}")

            # Thêm thông tin số lượng vi phạm
            draw_text(
                display_frame,
                "So luong vi pham: " + str(len(array_helmet_filter)),
                font_scale=0.7,
                pos=(10, 30),
                text_color=(255, 255, 255),
                text_color_bg=(0, 0, 0),
            )

        except Exception as e:
            print(f"Lỗi khi xử lý frame: {e}")
            # Tạo frame thông báo lỗi để trả về
            error_text = f"Loi xu ly: {str(e)}"
            cv2.putText(display_frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Trả về frame đã xử lý thông qua generator
        yield display_frame

    # Đóng video khi xử lý xong
    cap.release()
    print("Đã đóng video stream")


if __name__ == "__main__":
    # Thử nghiệm chạy hàm video_detect_helmet độc lập
    video_gen = video_detect_helmet("Videos/mainn.mp4")
    for frame in video_gen:
        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Nhấn q để thoát
            break
    cv2.destroyAllWindows()