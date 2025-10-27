# Assignment

## 1. What is the difference between `cv2.waitKey(0)` and `cv2.waitKey(1)`?

`cv2.waitKey(0)` waits indefinitely until any key is pressed. The program pauses and the displayed window stays open until a key event occurs.  
`cv2.waitKey(1)` waits for 1 millisecond for a key event, then continues execution.
---

## 2. How can you read an image in a different color format (RGB instead of BGR)?

```python
import cv2

image_bgr = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```

---

## 3. Write a Python script that takes user input to choose a transformation ("darken", "lighten", "invert") and applies it to an image.

```python
import cv2

img = cv2.imread('image.jpg')
choice = input("Choose transformation (darken, lighten, invert): ").lower()

if choice == "darken":
    result = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
elif choice == "lighten":
    result = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
elif choice == "invert":
    result = cv2.bitwise_not(img)
else:
    print("Invalid choice")
    exit()

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. Selective Thresholding with Background

When you make only a specific range of brightness values white and keep the rest unchanged, only parts of the image within that range will become white. This technique highlights certain intensity regions of the image.

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)

lower, upper = 100, 150
mask = cv2.inRange(img, lower, upper)

result = img.copy()
result[mask > 0] = 255

cv2.imshow("Original", img)
cv2.imshow("Selective Threshold", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. How can you create a custom filter (kernel) and apply it to an image using OpenCV in Python?

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

filtered = cv2.filter2D(img, -1, kernel)

cv2.imshow("Original", img)
cv2.imshow("Filtered", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 6. Write a Python code to apply the Derivative of Gaussian (DoG) filter on an image using OpenCV.

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)

blur1 = cv2.GaussianBlur(img, (5, 5), 1)
blur2 = cv2.GaussianBlur(img, (5, 5), 2)

dog = cv2.absdiff(blur1, blur2)

cv2.imshow("Original", img)
cv2.imshow("DoG Filter", dog)
cv2.waitKey(0)
cv2.destroyAllWindows()
```