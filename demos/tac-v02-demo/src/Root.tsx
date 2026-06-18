import "./index.css";
import { Composition } from "remotion";
import { MyComposition } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="TacV02Demo"
        component={MyComposition}
        durationInFrames={3000}
        fps={10}
        width={1280}
        height={720}
      />
    </>
  );
};
