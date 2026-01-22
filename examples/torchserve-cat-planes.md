Request Payload:
```json
{
  "inputs": [
    {
      "name": "image-0",
      "datatype": "BYTES",
      "shape": [
        1
      ],
      "data": [
        "<base64-encoded-image>"
      ],
      "parameters": {
        "action": "detect"
      }
    },
    {
      "name": "image-1",
      "datatype": "BYTES",
      "shape": [
        1
      ],
      "data": [
        "<base64-encoded-image>"
      ],
      "parameters": {
        "action": "detect"
      }
    }
  ]
}
```

Response Payload:
```json
{
  "model_name": "model",
  "model_version": "1.0.0",
  "id": "fbfc837c-3123-4261-90ae-e6744f66124e",
  "parameters": null,
  "outputs": [
    {
      "name": "image-0",
      "shape": [
        1
      ],
      "datatype": "BYTES",
      "parameters": null,
      "data": [
        {
          "detection_classes": [],
          "detection_scores": [],
          "oriented_detection_boxes": []
        }
      ]
    },
    {
      "name": "image-1",
      "shape": [
        1
      ],
      "datatype": "BYTES",
      "parameters": null,
      "data": [
        {
          "detection_classes": [
            "plane",
            "plane"
          ],
          "detection_scores": [
            0.8744706511497498,
            0.8613941669464111
          ],
          "oriented_detection_boxes": [
            {
              "cx": 0.32175594568252563,
              "cy": 0.6674760580062866,
              "w": 0.084013931453228,
              "h": 0.08624937385320663,
              "r": 1.5669623613357544
            },
            {
              "cx": 0.13600456714630127,
              "cy": 0.47356200218200684,
              "w": 0.094029501080513,
              "h": 0.08115722239017487,
              "r": 1.5425734519958496
            }
          ]
        }
      ]
    }
  ]
}
```
