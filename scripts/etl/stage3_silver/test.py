import h3

def test_h3_version_4_compatibility():
    print("--- Chạy bài Test: Tương thích H3 (Version 4.x) ---")
    # Tọa độ hợp lệ tại Tòa thị chính Chicago
    lat, lon = 41.8832, -87.6324
    resolution = 9
    
    try:
        # Gọi hàm latlng_to_cell của version 4.x
        h3_index = h3.latlng_to_cell(lat, lon, resolution)
        
        assert h3_index is not None, "Kết quả không được trả về Null"
        assert isinstance(h3_index, str), "Mã H3 phải là dạng chuỗi (String)"
        assert h3_index.startswith("8926"), "Mã H3 ở Chicago thường phải bắt đầu bằng '8926'"
        
        print(f"[PASSED] Xử lý an toàn. Tọa độ ({lat}, {lon}) -> H3: {h3_index}")
        
    except AttributeError:
        print("[FAILED] Lỗi phiên bản thư viện! Hãy chắc chắn bạn đã dùng h3.latlng_to_cell")
    except Exception as e:
        print(f"[FAILED] Gặp lỗi hệ thống: {e}")

if __name__ == "__main__":
    test_h3_version_4_compatibility()