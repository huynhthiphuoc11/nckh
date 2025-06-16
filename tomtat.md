---
title: Untitled

---


## 1. Phương pháp tiếp cận tổng thể

- **Mục tiêu:** Cung cấp hệ thống đề xuất việc làm cá nhân hóa dựa trên sở thích người dùng và dữ liệu lịch sử công việc.
- **Dữ liệu:** Thu thập dữ liệu việc làm từ Glassdoor, bao gồm các thuộc tính như tiêu đề công việc, mức lương, mô tả công việc, đánh giá công ty, vị trí, ngành nghề, v.v.
- **Triển khai:** Hệ thống được triển khai trên nền tảng đám mây Azure.

---

## 2. Các bước xử lý và mô hình

### a) Thu thập và tiền xử lý dữ liệu (Data Scraping & Feature Engineering)

- Thu thập dữ liệu việc làm từ Glassdoor qua scraping.
- Xử lý dữ liệu thiếu (imputation hoặc loại bỏ).
- Mã hóa biến phân loại (one-hot encoding, label encoding) cho các trường như tiêu đề công việc, vị trí, ngành nghề.
- Chuẩn hóa các biến số như mức lương, đánh giá công ty để cân bằng tầm ảnh hưởng.

### b) Mô hình học máy

- Dự án sử dụng các thuật toán học máy truyền thống để huấn luyện mô hình đề xuất.
- Mặc dù không nêu rõ thuật toán cụ thể trong repo, theo các nghiên cứu và dự án tương tự, các thuật toán phổ biến bao gồm:

  - **Random Forest (RF)**
  - **Support Vector Machine (SVM)**
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Naive Bayes**

- Các mô hình này được huấn luyện để dự đoán mức độ phù hợp của công việc dựa trên các đặc trưng đầu vào.

### c) Vector hóa và xử lý ngôn ngữ tự nhiên (NLP)

- Sử dụng kỹ thuật **TF-IDF vectorization** để chuyển đổi các trường văn bản như tiêu đề công việc, kỹ năng chính thành dạng số để mô hình học máy xử lý.
- Có thể kết hợp thêm các kỹ thuật NLP nâng cao để hiểu sâu hơn nội dung mô tả công việc và hồ sơ người dùng.

### d) Đề xuất dựa trên lọc cộng tác và cá nhân hóa

- Một số hệ thống đề xuất việc làm hiện đại kết hợp **collaborative filtering** để tận dụng lịch sử tương tác của người dùng.
- Ngoài ra, dựa trên hồ sơ kỹ năng, kinh nghiệm, ngành nghề để cá nhân hóa đề xuất.

---

## 3. Tổng hợp phương pháp áp dụng trong dự án này

| Thành phần                      | Phương pháp / Thuật toán áp dụng                   | Mô tả                                                                                   |
|--------------------------------|----------------------------------------------------|-----------------------------------------------------------------------------------------|
| Thu thập dữ liệu               | Web scraping Glassdoor                              | Thu thập thông tin việc làm đa dạng                                                    |
| Tiền xử lý dữ liệu             | Imputation, One-hot encoding, Label encoding, Scaling | Chuẩn hóa và chuyển đổi dữ liệu phù hợp cho mô hình                                    |
| Xử lý văn bản                 | TF-IDF vectorization                                | Biến đổi văn bản thành vector số phục vụ huấn luyện                                   |
| Mô hình học máy               | Random Forest, SVM, Logistic Regression, KNN, Naive Bayes (theo các dự án tương tự) | Dự đoán mức độ phù hợp công việc với người dùng                                        |
| Phương pháp đề xuất           | Collaborative Filtering (tham khảo các nghiên cứu tương tự) | Cá nhân hóa đề xuất dựa trên lịch sử và sở thích người dùng                            |
| Triển khai                    | Azure Cloud                                        | Hệ thống được triển khai trên nền tảng đám mây để phục vụ người dùng                   |

---

## 4. Tham khảo thêm từ các nghiên cứu liên quan

- Các nghiên cứu chuyên sâu về hệ thống đề xuất việc làm thường kết hợp:

  - **Collaborative Filtering** để tận dụng dữ liệu tương tác người dùng.
  - **Natural Language Processing (NLP)** để hiểu nội dung mô tả công việc và hồ sơ.
  - **Deep Learning (CNN, Attention Mechanism)** để nâng cao độ chính xác đề xuất (một số nghiên cứu mới).
  - **Random Forest và SVM** được sử dụng phổ biến để phân loại và dự đoán mức độ phù hợp công việc với người dùng dựa trên đặc trưng cá nhân.

---

## Kết luận

Dự án "Job Recommendation System" chủ yếu áp dụng **machine learning truyền thống** kết hợp với **kỹ thuật xử lý dữ liệu và NLP cơ bản (TF-IDF)** để xây dựng mô hình đề xuất việc làm cá nhân hóa. Thuật toán cụ thể có thể bao gồm Random Forest, SVM hoặc các mô hình phân loại khác, cùng với các bước tiền xử lý dữ liệu kỹ lưỡng nhằm tối ưu hiệu quả đề xuất.




-----------------------------------------------------------
Bài báo "Embedding-based Recommender System for Job to Candidate Matching on Scale" trình bày một hệ thống đề xuất việc làm quy mô lớn sử dụng phương pháp **hệ thống đề xuất hai giai đoạn (two-stage recommender system)** kết hợp với kỹ thuật **embedding học sâu (deep learning embedding)** để khớp nối hiệu quả giữa tin tuyển dụng và ứng viên. Dưới đây là phân tích chi tiết về thuật toán và phương pháp tiếp cận họ sử dụng:

---

## 1. Kiến trúc hệ thống hai giai đoạn

- **Giai đoạn 1: Candidate Retrieval (Tìm kiếm ứng viên tiềm năng)**
  - Sử dụng mô hình embedding hai tháp (two-tower embedding model) để học biểu diễn (embedding) riêng biệt cho tin tuyển dụng và hồ sơ ứng viên.
  - Mỗi tháp (tower) là một mạng neural nhận đầu vào là dữ liệu văn bản thô (mô tả công việc, hồ sơ ứng viên) và các đặc trưng bổ sung.
  - Biểu diễn embedding của công việc và ứng viên được tính toán độc lập, sau đó dùng để tìm kiếm gần nhất xấp xỉ (approximate nearest neighbor search) nhằm lọc ra hàng trăm ứng viên tiềm năng từ hàng triệu hồ sơ.
  - Sử dụng thư viện **Faiss** để xây dựng chỉ mục (index) embedding và thực hiện tìm kiếm nhanh trong không gian embedding.

- **Giai đoạn 2: Re-ranking (Sắp xếp lại kết quả)**
  - Sau khi thu hẹp danh sách ứng viên tiềm năng, một mô hình sắp xếp lại (reranking) sử dụng các đặc trưng ngữ cảnh bổ sung (ví dụ: kỹ năng, kinh nghiệm, vị trí địa lý) để đánh giá và xếp hạng chính xác hơn.
  - Mô hình này giúp tinh chỉnh điểm số khớp nối, chọn ra số lượng ứng viên phù hợp nhất để đề xuất.

---

## 2. Mô hình embedding học sâu (Deep Learning Embedding Model - DLEM)

- **Input:** Cặp dữ liệu (job post, candidate resume) dưới dạng văn bản thô.
- **Kiến trúc:**  
  - Lớp đầu vào (input layer) nhận chuỗi văn bản.  
  - Lớp CNN (Convolutional Neural Network) với nhiều bộ lọc kích thước khác nhau để trích xuất đặc trưng ngữ nghĩa đa cấp độ từ văn bản.  
  - Lớp attention giúp tập trung vào các phần quan trọng trong văn bản, nâng cao chất lượng biểu diễn.  
- **Huấn luyện:**  
  - Mô hình được huấn luyện theo dạng học có giám sát (supervised learning) dựa trên dữ liệu ứng viên đã ứng tuyển (positive pairs) và các cặp tiêu cực được chọn lọc.  
  - Mục tiêu là học embedding sao cho khoảng cách giữa embedding của công việc và ứng viên phù hợp là nhỏ, còn với các cặp không phù hợp là lớn.

---

## 3. Fused Embedding (Biểu diễn hợp nhất)

- Kết hợp ba loại embedding để tạo biểu diễn tổng thể cho công việc và ứng viên:
  - **Embedding văn bản thô:** từ mô tả công việc và hồ sơ ứng viên qua mô hình CNN + attention.
  - **Embedding ngữ nghĩa từ đồ thị kỹ năng và công việc:** khai thác mối quan hệ giữa các kỹ năng, công việc, lịch sử chuyển đổi nghề nghiệp.
  - **Embedding vị trí địa lý:** chuyển tọa độ địa lý thành vector không gian 3 chiều để phản ánh vị trí làm việc và ứng viên.

---

## 4. Tìm kiếm gần nhất xấp xỉ (Approximate Nearest Neighbor Search)

- Sử dụng **Faiss index** để lưu trữ embedding của ứng viên và thực hiện tìm kiếm nhanh trong không gian vector.
- Giúp hệ thống có thể xử lý quy mô hàng triệu hồ sơ và trả về danh sách ứng viên tiềm năng trong thời gian thực.

---

## 5. Đánh giá và hiệu quả

- **Đánh giá offline:**  
  - So sánh với hệ thống baseline dùng Apache Solr (dựa trên phân loại phân cấp và nội dung).  
  - Hệ thống mới cải thiện khoảng 19% điểm chất lượng và 18% nDCG (Normalized Discounted Cumulative Gain).  
- **Đánh giá online:**  
  - Dựa trên hơn 120.000 lượt hiển thị và click trong 4 tháng.  
  - CTR (Click Through Rate) tăng khoảng 104%, nDCG tăng khoảng 37%.  
- Kết quả cho thấy hệ thống hai giai đoạn với embedding hợp nhất vượt trội so với các phương pháp truyền thống.

---

## 6. Tổng kết ưu điểm

- **Khả năng mở rộng:** Xử lý hiệu quả hàng triệu đến hàng tỷ hồ sơ nhờ embedding và Faiss.
- **Giảm vấn đề cold-start:** Embedding dựa trên nội dung giúp đề xuất ngay cả khi không có dữ liệu tương tác lịch sử.
- **Đa nguồn dữ liệu:** Kết hợp văn bản, kỹ năng, vị trí địa lý giúp biểu diễn toàn diện hơn.
- **Hai giai đoạn:** Tách biệt tìm kiếm nhanh và sắp xếp chính xác giúp cân bằng hiệu năng và chất lượng.

---

# Tóm tắt

| Thành phần                      | Phương pháp / Thuật toán                         | Mô tả                                                                                   |
|--------------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------------|
| Kiến trúc hệ thống             | Hai giai đoạn (retrieval + reranking)            | Tìm kiếm ứng viên tiềm năng, sau đó sắp xếp lại chính xác                               |
| Mô hình embedding              | Deep Learning Embedding Model (CNN + Attention) | Học biểu diễn ngữ nghĩa từ văn bản và dữ liệu bổ sung                                  |
| Biểu diễn hợp nhất (Fused Embedding) | Kết hợp embedding văn bản, đồ thị kỹ năng, vị trí | Tạo biểu diễn toàn diện cho công việc và ứng viên                                      |
| Tìm kiếm gần nhất xấp xỉ       | Faiss index                                      | Tìm kiếm nhanh trong không gian embedding quy mô lớn                                   |
| Huấn luyện                    | Học có giám sát với dữ liệu cặp (job, candidate) | Tối ưu embedding để phản ánh mức độ phù hợp                                            |
| Đánh giá                      | Offline (quality score, nDCG), Online (CTR, nDCG) | Cải thiện đáng kể so với baseline truyền thống                                         |
