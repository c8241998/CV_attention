class CBAM_Module(fluid.dygraph.Layer):
    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(pool_size=1, pool_type="avg")
        self.max_pool = AdaptiveAvgPool2d(pool_size=1, pool_type="max")
        self.fc1 = fluid.dygraph.Conv2D(num_channels=channels, num_filters=channels // reduction, filter_size=1, padding=0)
        self.relu = ReLU()
        self.fc2 = fluid.dygraph.Conv2D(num_channels=channels // reduction, num_filters=channels, filter_size=1, padding=0)

        self.sigmoid_channel = Sigmoid()
        self.conv_after_concat = fluid.dygraph.Conv2D(num_channels=2, num_filters=1, filter_size=7, stride=1, padding=3)
        self.sigmoid_spatial = Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        module_input = x
        avg = self.relu(self.fc1(self.avg_pool(x)))
        avg = self.fc2(avg)
        mx = self.relu(self.fc1(self.max_pool(x)))
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # Spatial Attention Module
        x = module_input * x
        module_input = x
        avg = fluid.layers.mean(x)
        mx = fluid.layers.argmax(x, axis=1)
        print(avg.shape, mx.shape)
        x = fluid.layers.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x

        return x