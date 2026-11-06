# Báo cáo cá nhân — Day 01 Lab: AI Foundation & LLM

- **Họ và tên:** Dương Ngọc Tuyên
- **MSSV:** 2A202603511
- **Bài thực hành:** Day 01 — Một bài toán, bốn thế hệ AI: Phân loại SMS Spam / Ham
- **Notebook:** `day01_lab.ipynb`

---

## 1. Mục tiêu và nội dung bài lab

Trong bài lab, em thực hiện bài toán phân loại tin nhắn SMS thành hai nhãn:

- `spam`: Tin nhắn rác, quảng cáo không mong muốn hoặc có dấu hiệu lừa đảo.
- `ham`: Tin nhắn bình thường.

Mục tiêu là giải cùng một bài toán bằng các cách tiếp cận khác nhau, từ Machine Learning truyền thống, Deep Learning, Embedding đến Large Language Model (LLM). Sau đó, em so sánh chất lượng dự đoán, thời gian huấn luyện, tốc độ suy luận, độ trễ và chi phí để xem phương pháp nào phù hợp hơn khi triển khai thực tế.

Các phần chính em đã thực hiện:

1. Khám phá và tiền xử lý dữ liệu SMS.
2. Tự cài đặt mô hình Logistic Regression bằng NumPy.
3. Xây dựng và huấn luyện mạng MLP bằng PyTorch.
4. Sử dụng MiniLM tạo Embedding, sau đó thử kNN, Logistic Regression và MLP head.
5. Thực hành phân loại bằng LLM thông qua prompt zero-shot và few-shot; kiểm tra trạng thái API.
6. Tổng hợp benchmark, phân tích Pareto, stress test và thực hiện thử thách cá nhân về kích thước Vocabulary.

## 2. Phần 0 — Khám phá và chuẩn bị dữ liệu

### 2.1. Nạp và quan sát dữ liệu

Em sử dụng bộ dữ liệu **SMS Spam Collection** từ Kaggle (`uciml/sms-spam-collection-dataset`).

Kết quả nạp dữ liệu:

| Nội dung | Số lượng |
|---|---:|
| Tổng số tin nhắn ban đầu | 5.572 |
| Tin nhắn `ham` | 4.825 |
| Tin nhắn `spam` | 747 |
| Tin nhắn trùng lặp | 403 |
| Tổng số tin sau khi bỏ trùng | 5.169 |

Em kiểm tra độ dài tin nhắn theo từng nhãn. Độ dài trung bình của tin spam là khoảng 23,85 từ, trong khi ham khoảng 14,20 từ. Kết quả này cho thấy hai nhóm có một số khác biệt về độ dài, nhưng độ dài không đủ để tự nó xác định một tin nhắn có phải spam hay không.

### 2.2. Loại bỏ dữ liệu trùng lặp và chia tập

Em loại bỏ 403 tin nhắn trùng lặp trước khi chia dữ liệu để hạn chế **data leakage** — trường hợp nội dung giống nhau xuất hiện đồng thời trong train và test, khiến kết quả đánh giá có thể cao hơn thực tế.

Sau khi loại trùng, dữ liệu được chia như sau:

| Tập dữ liệu | Số tin nhắn | Số tin spam |
|---|---:|---:|
| Train | 3.619 | 462 |
| Validation | 775 | — |
| Test | 775 | 96 |
| SUB | 40 | Tập con dùng để so sánh LLM |

Việc chia dữ liệu được tạo theo `STUDENT_ID` thông qua seed riêng của notebook. Với MSSV `2A202603511`, notebook ghi nhận `SEED=6847`.

Em cũng nhận thấy dữ liệu bị mất cân bằng: số tin ham lớn hơn nhiều so với số tin spam. Vì vậy, ngoài Accuracy, cần quan sát Precision, Recall và F1 để đánh giá mô hình đầy đủ hơn.

## 3. Phần 1 — Machine Learning: Bag-of-Words và Logistic Regression

### 3.1. Vector hóa văn bản

Em xây dựng Vocabulary gồm tối đa 2.000 từ từ tập train. Mỗi SMS được chuyển thành vector Bag-of-Words (BoW), trong đó mỗi vị trí biểu diễn một từ trong Vocabulary và cho biết từ đó có xuất hiện trong tin nhắn hay không.

Ví dụ, câu `"WIN a FREE prize"` được tách thành các token rồi biểu diễn thành vector đặc trưng. Cách biểu diễn này giúp mô hình xử lý văn bản dưới dạng số, nhưng không giữ được đầy đủ thứ tự từ và ngữ cảnh câu.

Notebook kiểm tra thành công các hàm:

- `build_vocab(...)`: xây dựng từ điển từ tập train.
- `vectorize(...)`: chuyển văn bản thành vector BoW.
- `sigmoid(...)` và `predict_proba(...)`: tính xác suất thuộc lớp spam.
- `metrics(...)`: tính các chỉ số đánh giá.

### 3.2. Tự cài đặt Logistic Regression

Em tự triển khai Logistic Regression bằng NumPy, không sử dụng `sklearn` trong các hàm tự viết. Mô hình tính xác suất spam bằng hàm sigmoid:

\[
p(\text{spam}\mid x)=\sigma(w\cdot x+b)
\]

Quá trình huấn luyện sử dụng Gradient Descent. Ở mỗi epoch, mô hình tính xác suất dự đoán, tính sai số giữa dự đoán và nhãn thật, tính gradient cho trọng số và bias, sau đó cập nhật tham số theo learning rate.

Kết quả trên tập test:

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 0,973 |
| Precision | 0,987 |
| Recall | 0,792 |
| F1 | 0,879 |
| Thời gian train | 3,47 giây |
| Throughput benchmark | 11.230 tin/giây |
| p50 | 0,057 ms/tin |
| p95 | 0,107 ms/tin |

Precision cao nhưng Recall thấp hơn. Điều này cho thấy mô hình dự đoán spam khá thận trọng: khi đã dự đoán spam thì thường đúng, nhưng vẫn bỏ sót một số tin spam.

Notebook cũng in ra các từ có trọng số đẩy dự đoán về spam, ví dụ `call`, `txt`, `free`, `text`, `stop`, `reply`, `now`; và các từ kéo dự đoán về ham, ví dụ `i`, `me`, `my`, `that`, `it`.

### 3.3. Thử nghiệm Learning Rate và kiểm tra lỗi phương pháp

Em thử thay đổi learning rate, gồm `0.001` và `50`, để quan sát ảnh hưởng của tốc độ cập nhật trọng số. Learning rate quá nhỏ làm loss giảm chậm; learning rate quá lớn có thể khiến các bước cập nhật vượt quá điểm tối ưu, làm quá trình học dao động hoặc khó ổn định.

Em phân tích đoạn code có Accuracy cao nhưng mắc lỗi phương pháp:

- Huấn luyện và đánh giá trên cùng dữ liệu, không có tập test độc lập.
- Dùng dữ liệu gốc chưa loại trùng, làm tăng nguy cơ data leakage.

Qua phần này, em hiểu rằng Accuracy cao không đủ chứng minh mô hình tổng quát hóa tốt nếu quy trình chia dữ liệu và đánh giá không đúng.

## 4. Phần 2 — Deep Learning: MLP bằng PyTorch

### 4.1. Xây dựng mạng MLP

Em xây dựng mạng MLP nhận đầu vào là vector BoW 2.000 chiều, có một hidden layer 64 nơ-ron, hàm kích hoạt ReLU và lớp đầu ra để dự đoán spam/ham.

Với kiến trúc `build_mlp(2000, 64)`, notebook ghi nhận **128.129 tham số**, lớn hơn đáng kể so với Logistic Regression có 2.001 tham số.

### 4.2. Huấn luyện và đánh giá

Em tự triển khai vòng lặp huấn luyện bằng PyTorch với các bước chính:

1. `opt.zero_grad()`: xóa gradient từ vòng lặp trước.
2. `model(xb)`: thực hiện forward pass.
3. `loss_fn(logits, yb)`: tính hàm mất mát.
4. `loss.backward()`: backpropagation để tính gradient.
5. `opt.step()`: cập nhật tham số của mô hình.

Em theo dõi train loss và validation loss trong 30 epoch. Validation loss thấp nhất tại epoch 5; sau đó train loss tiếp tục giảm nhưng validation loss có xu hướng tăng, là dấu hiệu mô hình bắt đầu overfit. Vì vậy, epoch 5 là điểm dừng phù hợp theo kết quả quan sát.

Kết quả MLP trên tập test:

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 0,983 |
| Precision | 0,966 |
| Recall | 0,896 |
| F1 | 0,930 |
| Thời gian train | 11,07 giây |
| Throughput benchmark | 26.272 tin/giây |
| p50 | 0,301 ms/tin |
| p95 | 0,493 ms/tin |

MLP đạt F1 cao hơn Logistic Regression trong thí nghiệm này. Tuy nhiên, mô hình có nhiều tham số và thời gian train dài hơn. Chênh lệch không có nghĩa Deep Learning luôn tốt hơn trong mọi bài toán; kết quả còn phụ thuộc dữ liệu và cách biểu diễn đầu vào.

### 4.3. Thử nghiệm bỏ ReLU

Em chạy lại mạng khi bỏ ReLU. Kết quả F1 test là **0,924**, thấp hơn kết quả 0,930 khi có ReLU. Hai lớp Linear liên tiếp nếu không có hàm kích hoạt phi tuyến có thể gộp thành một phép biến đổi tuyến tính, khiến mạng mất khả năng học quan hệ phi tuyến giữa các đặc trưng.

## 5. Phần 3 — Embedding với MiniLM

### 5.1. Tạo vector ngữ nghĩa

Em sử dụng encoder `sentence-transformers/all-MiniLM-L6-v2` để chuyển mỗi SMS thành vector embedding 384 chiều. Encoder đã được pre-train và được đóng băng; notebook không huấn luyện lại toàn bộ encoder.

Kết quả encode tập train:

- Kích thước ma trận embedding: `3619 × 384`.
- Thời gian encode tập train: khoảng 100,1 giây.
- Tốc độ encode trong lần chạy: khoảng 36 tin/giây.

Khác với BoW, embedding biểu diễn ngữ nghĩa của toàn câu. Vì vậy, các câu dùng từ khác nhau nhưng có ý nghĩa gần nhau có thể có vector gần nhau.

### 5.2. Cosine Similarity và tìm kiếm ngữ nghĩa

Em cài đặt phép đo Cosine Similarity và hàm tìm các tin nhắn gần nghĩa nhất. Với truy vấn:

`you have been selected to receive a complimentary holiday, reply now to claim`

các kết quả gần nhất đều là tin spam về trúng thưởng, kỳ nghỉ và nhận tiền. Điều này cho thấy embedding có thể tìm được nội dung tương đồng về nghĩa ngay cả khi truy vấn không dùng đúng các từ khóa spam quen thuộc.

Em cũng kiểm tra và giải thích hai lỗi trong hàm tìm kiếm tương tự: chưa chuẩn hóa vector trước khi tính cosine và sắp xếp tăng dần khiến hàm lấy các kết quả có điểm tương đồng thấp nhất.

### 5.3. Thử nghiệm ba bộ phân loại trên cùng encoder

Em so sánh ba cách sử dụng embedding:

- **3a. Encoder + kNN:** lưu vector train và dự đoán bằng cách lấy phiếu từ 5 mẫu gần nhất.
- **3b. Encoder + Logistic Regression:** dùng Logistic Regression làm đầu phân loại trên embedding.
- **3c. Encoder + MLP head:** dùng MLP có hidden layer 64 nơ-ron làm đầu phân loại.

Kết quả:

| Phương pháp | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Encoder + kNN | 0,975 | 0,923 | 0,875 | 0,898 |
| Encoder + LogReg | 0,983 | 0,956 | 0,906 | 0,930 |
| Encoder + MLP head | 0,988 | 0,978 | 0,927 | 0,952 |

Thời gian và tốc độ benchmark:

| Phương pháp | Train/encode (s) | Tin/giây | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|
| Encoder + kNN | 100,13 | 23,62 | 52,738 | 127,662 |
| Encoder + LogReg | 100,62 | 30,39 | 30,236 | 59,441 |
| Encoder + MLP head | 101,14 | 31,58 | 35,647 | 56,603 |

Thời gian của các phương pháp embedding bao gồm bước encode dữ liệu train; encoder MiniLM được đóng băng. Với kNN, không có bước tối ưu trọng số như LogReg/MLP mà mô hình lưu các vector train để truy vấn.

### 5.4. Kiểm tra các tin nhắn ngụy trang

Em thử sáu tin nhắn có từ khóa gây nhiễu hoặc spam né từ khóa. Kết quả:

- BoW dự đoán đúng 5/6 tin.
- Encoder + Logistic Regression dự đoán đúng 6/6 tin.

Ví dụ, BoW bỏ sót tin nhắn có nội dung cảnh báo tài khoản sắp hết hạn và yêu cầu xác nhận thông tin. Encoder + LogReg nhận diện đúng hơn nhờ biểu diễn ngữ nghĩa toàn câu thay vì chỉ dựa vào các từ khóa riêng lẻ.

## 6. Phần 4 — LLM: phân loại bằng Prompt

### 6.1. Nội dung đã chuẩn bị trong notebook

Notebook thiết kế phần LLM để gọi API và phân loại trên tập SUB gồm 40 tin, nhằm hạn chế chi phí. Các nội dung được triển khai/kiểm tra trong code gồm:

- Hàm tạo prompt phân loại SMS.
- Hàm parse kết quả trả về thành nhãn spam/ham.
- Hàm gọi API theo template `call_openai`, `call_openai_mini` và cơ chế retry.
- Hàm phân loại theo batch.
- Thí nghiệm so sánh model lớn và model nhỏ theo zero-shot.
- Thí nghiệm few-shot bằng cách đưa thêm 4 ví dụ từ tập train vào prompt.

Notebook cũng có cấu hình lựa chọn provider như OpenAI, Anthropic, Gemini, Groq, OpenRouter hoặc Ollama.

### 6.2. Trạng thái chạy thực tế

Trong lần chạy notebook được dùng để lập báo cáo, ô nhập `GROQ_API_KEY` ghi nhận **không có key** và thông báo các ô gọi API sẽ được bỏ qua. Vì vậy, các thí nghiệm gọi LLM (zero-shot/few-shot) không có kết quả API thực tế trong output của notebook này. Không sử dụng các con số F1 hoặc chi phí LLM giả định để coi là kết quả đã đo.

Em vẫn phân tích được các vấn đề trong thiết kế:

- `temperature=0` giúp đầu ra ổn định hơn khi chỉ cần nhãn.
- Giới hạn `max_tokens` giúp tránh câu trả lời dài không cần thiết, nhưng giới hạn quá thấp có thể làm đầu ra bị cắt cụt.
- Hàm parse chỉ kiểm tra `"spam" in response.lower()` có thể sai nếu phản hồi chứa cụm phủ định như “not spam”.
- Few-shot có thể cung cấp ví dụ định hướng cho mô hình, nhưng làm tăng số token input và chi phí.

## 7. Phần 5 — Tổng hợp benchmark và phân tích Pareto

### 7.1. Bảng so sánh các mô hình

Bảng dưới đây là kết quả benchmark được in trực tiếp từ notebook:

| Phương pháp | F1 test | F1 SUB (40) | Train (s) | Tin/s | p50 (ms) | p95 (ms) | USD/1k tin |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoW + LogReg | 0,879 | 0,824 | 3,471 | 11.229,91 | 0,057 | 0,107 | 0 |
| BoW + MLP | 0,930 | 0,947 | 11,074 | 26.271,85 | 0,301 | 0,493 | 0 |
| MiniLM + kNN | 0,898 | 0,919 | 100,127 | 23,62 | 52,738 | 127,662 | 0 |
| MiniLM + LogReg | 0,930 | 0,947 | 100,615 | 30,39 | 30,236 | 59,441 | 0 |
| MiniLM + MLP head | 0,952 | 0,919 | 101,138 | 31,58 | 35,647 | 56,603 | 0 |

Lưu ý: LLM chỉ được thiết kế để đánh giá trên tập SUB 40 tin. Do API không có key trong lần chạy này, bảng benchmark không có hàng kết quả LLM thực nghiệm. Giá trị USD bằng 0 ở các hàng mô hình cục bộ, không có nghĩa chi phí hạ tầng triển khai bằng 0.

### 7.2. Đường Pareto và SLA

Notebook sử dụng biểu đồ đánh đổi giữa `1 − F1` và p95 latency. Kết quả ghi nhận hai phương pháp nằm trên đường Pareto:

- BoW + Logistic Regression.
- BoW + MLP.

Với ngưỡng SLA `p95 < 50 ms`, hai phương pháp BoW đều đạt yêu cầu. Trong nhóm này, BoW + MLP có F1 test cao hơn (0,930 so với 0,879), trong khi p95 vẫn thấp (0,493 ms).

Các phương pháp MiniLM có chất lượng tốt, đặc biệt MiniLM + MLP head đạt F1 test 0,952, nhưng p95 từ 56,603 ms trở lên nên không đạt ngưỡng 50 ms trong benchmark này.

### 7.3. Lựa chọn cho hệ thống lọc 10 triệu SMS/ngày

Với 10 triệu SMS/ngày, tốc độ trung bình cần xử lý là khoảng 116 tin/giây. Dựa trên throughput benchmark, BoW + LogReg và BoW + MLP đều cao hơn mức trung bình này. Trong hai phương pháp, BoW + MLP là lựa chọn cân bằng hơn vì có F1 cao hơn đáng kể và vẫn đáp ứng p95 dưới 50 ms.

Đây là kết luận dựa trên benchmark trong notebook; khi triển khai thật cần đo lại trên phần cứng production, tải đồng thời, batching, tài nguyên CPU/RAM và các điều kiện vận hành thực tế.

### 7.4. Ước tính chi phí LLM

Phần tính tay của notebook dùng giả định:

- 1.000.000 SMS/ngày.
- Mỗi SMS: 100 token input và 1 token output.
- Giá input: 0,10 USD/1 triệu token.
- Giá output: 0,50 USD/1 triệu token.
- Không tính token suy luận.

Kết quả kiểm tra tính tay trong notebook: **10,50 USD/ngày** cho 1 triệu SMS.

Nếu tăng lên 10 triệu SMS/ngày và giữ nguyên giả định token/giá:

- Chi phí ước tính: 105 USD/ngày.
- Chi phí 30 ngày: **3.150 USD/tháng**.

Đây là ước tính theo giả định của bài lab, không phải báo giá API hiện hành; chưa tính retry, prompt dài hơn, token phát sinh hoặc chi phí hạ tầng.

### 7.5. Đề xuất Router kết hợp ML và LLM

Em đề xuất kiến trúc lai gồm mô hình ML xử lý trước và LLM xử lý các trường hợp không chắc chắn:

1. SMS đi vào mô hình BoW (có thể chọn BoW + MLP làm mô hình chính).
2. Nếu xác suất spam `p < 0,3`, mô hình phân loại là ham.
3. Nếu `p > 0,7`, mô hình phân loại là spam.
4. Nếu `0,3 ≤ p ≤ 0,7`, chuyển tin nhắn sang LLM để đánh giá bổ sung.

Cách kết hợp này giúp phần lớn tin nhắn được xử lý bằng mô hình cục bộ có độ trễ thấp, giảm số lượt gọi LLM và tận dụng khả năng phân tích ngữ cảnh của LLM đối với các tin khó. Tuy nhiên, cần đo tỷ lệ tin đi qua nhánh LLM, đặt timeout/fallback và kiểm tra lại p95 toàn hệ thống để bảo đảm SLA.

## 8. Stress test — 50 trường hợp khó

### 8.1. Mục đích và nhóm dữ liệu

Ngoài tập test thông thường, notebook tạo bộ 50 case khó viết tay để kiểm tra độ bền vững của mô hình trước những tình huống dễ gây nhầm lẫn:

| Nhóm | Số case | Nội dung |
|---|---:|---|
| Ham trông giống spam | 10 | Tin bình thường có từ khóa như FREE, WON, URGENT |
| Spam giọng tự nhiên | 10 | Spam không dùng từ khóa quảng cáo quen thuộc |
| Thao túng/lừa đảo | 8 | Mạo danh, gây sợ hãi hoặc tạo áp lực thời hạn |
| Né từ khóa | 6 | Thay ký tự, chèn dấu hoặc khoảng trắng |
| Prompt injection | 6 | 5 spam và 1 ham, có nội dung chỉ dẫn đánh lừa LLM |
| Tin giao dịch hợp lệ | 6 | OTP, lịch hẹn, đơn hàng |
| Nói về spam | 4 | Tin bình thường đang kể lại/cảnh báo spam |

### 8.2. Kết quả stress test

| Mô hình | F1 test thường (SUB) | F1 50 case khó | Accuracy 50 case |
|---|---:|---:|---:|
| BoW + LogReg | 0,824 | 0,378 | 0,54 |
| BoW + MLP | 0,947 | 0,450 | 0,56 |
| Embedding + kNN | 0,919 | 0,750 | 0,72 |
| Embedding + LogReg | 0,947 | 0,667 | 0,66 |
| Embedding + MLP head | 0,919 | 0,612 | 0,62 |

Kết quả cho thấy F1 trên tập test thông thường cao không đồng nghĩa mô hình xử lý tốt mọi tình huống. Các mô hình BoW giảm mạnh trên bộ case khó vì phụ thuộc nhiều vào từ khóa. Embedding + kNN có F1 cao nhất trong stress test này (0,750), nhưng kết quả chỉ phản ánh bộ case được viết tay, không phải benchmark đại diện cho toàn bộ SMS ngoài đời.

Notebook ghi nhận có 6 case mà tất cả mô hình đều dự đoán sai; phần lớn là spam giọng tự nhiên, ngoài ra có một tin OTP hợp lệ. Điều này cho thấy việc nhận diện ý định và ngữ cảnh vẫn là thách thức.

### 8.3. Prompt injection và phòng thủ

Notebook có hàm `build_defended_prompt(...)`, đặt nội dung SMS trong thẻ `<sms>...</sms>` và nhấn mạnh rằng mọi nội dung bên trong là dữ liệu không đáng tin cậy, không phải chỉ dẫn dành cho mô hình.

Ngoài phòng thủ bằng prompt, có thể kết hợp chuẩn hóa đầu vào, phát hiện ký tự bị biến đổi, rule-based filtering và một tầng kiểm tra độc lập. Không nên để LLM là lớp bảo vệ duy nhất.

Bộ 50 case khó phù hợp để stress test, tìm điểm yếu và kiểm tra robustness; không nên dùng riêng bộ này để ước lượng hiệu năng trung bình trên lưu lượng SMS thực tế vì phân phối case đã được cố ý làm khó.

### 8.4. Bổ sung case cá nhân

Em đã thêm 5 case tự nghĩ ra vào `MY_CASES`, gồm:

- Tin ham có nội dung vay tiền gấp nhưng là trao đổi giữa bạn bè.
- Spam giọng tự nhiên quảng bá cơ hội đầu tư.
- Tin lừa đảo giả danh shipper và yêu cầu chuyển tiền.
- Spam né từ khóa bằng cách thay ký tự.
- Tin prompt injection bằng tiếng Việt.

Trong output notebook, ô thêm case ghi nhận đã thêm 5 case. Tuy nhiên, lần chạy được lưu không thể hiện kết quả chạy lại toàn bộ mô hình trên bộ dữ liệu đã bổ sung 5 case này; do đó không đưa ra kết luận về độ chính xác của các case cá nhân.

## 9. Thử thách cá nhân — Đề #2: Thay đổi kích thước Vocabulary

Đề được notebook chọn theo MSSV là:

**ML — Giảm Vocabulary còn 100 từ và tăng lên 5.000 từ bằng `build_vocab(..., max_size)`. Huấn luyện lại và báo F1 test.**

Em giữ nguyên dữ liệu train/test, thuật toán Logistic Regression, learning rate `0.5` và số epoch `300`; chỉ thay đổi giới hạn số từ trong Vocabulary.

### 9.1. Kết quả thực nghiệm

| Vocabulary tối đa | Số từ thực tế | Precision | Recall | F1 test |
|---:|---:|---:|---:|---:|
| 100 | 100 | 0,944 | 0,698 | 0,802 |
| 5.000 | 5.000 | 0,987 | 0,781 | 0,872 |

Khi tăng Vocabulary từ 100 lên 5.000 từ:

- Precision tăng từ 0,944 lên 0,987.
- Recall tăng từ 0,698 lên 0,781.
- F1 tăng từ 0,802 lên 0,872, tức tăng 0,070 điểm F1.

### 9.2. Giải thích

Với Vocabulary chỉ có 100 từ, mô hình bị giới hạn số đặc trưng nên có thể không giữ được một số từ quan trọng để nhận diện spam. Điều này thể hiện rõ ở Recall thấp hơn: mô hình bỏ sót nhiều tin spam hơn.

Khi tăng lên 5.000 từ, mô hình có thêm đặc trưng để nhận diện các cách diễn đạt đa dạng hơn. Trong lần chạy này, cả Precision, Recall và F1 đều tăng.

Tuy nhiên, không thể kết luận rằng Vocabulary càng lớn thì F1 luôn tăng. Nếu bổ sung quá nhiều từ ít xuất hiện hoặc không liên quan đến nhãn, mô hình có thể học thêm nhiễu và tăng độ phức tạp. Cần đánh giá bằng tập validation/test để chọn kích thước phù hợp.

## 10. Bài tính tay

Notebook ghi nhận cả 5 kết quả tính tay đều đạt kiểm tra:

| Nội dung | Kết quả |
|---|---:|
| Baseline Accuracy của mô hình luôn đoán ham | 0,87 |
| Sigmoid tại 0 | 0,50 |
| F1 với TP = 80, FP = 20, FN = 40 | 0,73 |
| Cosine giữa `[1, 2, 0]` và `[2, 1, 2]` | 0,60 |
| Chi phí LLM cho 1 triệu SMS/ngày theo giả định bài lab | 10,50 USD/ngày |

## 11. Kết luận và điều rút ra

Qua bài lab, em đã thực hành nhiều cách tiếp cận cho cùng một bài toán phân loại SMS:

- **Machine Learning:** tự xây dựng Vocabulary, vector hóa BoW và cài đặt Logistic Regression bằng NumPy.
- **Deep Learning:** xây dựng MLP, huấn luyện bằng PyTorch, theo dõi loss và nhận biết dấu hiệu overfitting.
- **Embedding:** sử dụng MiniLM tạo vector ngữ nghĩa, tính Cosine Similarity, tìm kiếm theo nghĩa và thử nghiệm kNN/LogReg/MLP head.
- **LLM:** chuẩn bị quy trình prompt zero-shot/few-shot, parse nhãn và gọi API; tuy nhiên lần chạy này không có API key nên chưa có kết quả LLM thực nghiệm.
- **Đánh giá hệ thống:** so sánh F1, thời gian train, throughput, p95 latency, phân tích Pareto, stress test và ước tính chi phí.

Kết quả cho thấy mô hình có F1 cao nhất trên test thông thường không nhất thiết là lựa chọn phù hợp nhất cho sản phẩm. MiniLM + MLP head đạt F1 0,952 nhưng p95 là 56,603 ms, vượt ngưỡng 50 ms. BoW + MLP đạt F1 0,930, p95 0,493 ms và throughput khoảng 26.272 tin/giây, nên là phương án cân bằng hơn trong benchmark hiện tại.

Điều em rút ra là cần đánh giá mô hình trên nhiều khía cạnh: chất lượng dự đoán, độ trễ, năng lực xử lý, chi phí và độ bền vững trước dữ liệu khó. Không nên chỉ dựa vào một chỉ số F1 hoặc Accuracy để quyết định mô hình đưa vào thực tế.

